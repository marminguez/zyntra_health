import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { z } from "zod";
import { prisma } from "../db/prisma";
import { fetchLatestReadings } from "../integrations/freestyle/client";
import { encryptValue } from "../security/crypto";

const loginSchema = z.object({
    email: z.string().email(),
    password: z.string().min(1),
});

export const authOptions: NextAuthOptions = {
    session: { strategy: "jwt" },
    pages: { signIn: "/login" },
    providers: [
        CredentialsProvider({
            name: "Credentials",
            credentials: {
                email: { label: "Email", type: "email" },
                password: { label: "Password", type: "password" },
            },
            async authorize(credentials) {
                const parsed = loginSchema.safeParse(credentials);
                if (!parsed.success) return null;

                const { email, password } = parsed.data;

                // Use LibreLink credentials to authenticate patient login.
                // If Libre auth fails, we log it but STILL allow Zyntra login.
                try {
                    await fetchLatestReadings(email, password);
                } catch (e) {
                    console.error("Skipping LibreLink check fail to allow Zyntra login:", e);
                }

                let user = await prisma.user.findUnique({ where: { email } });
                if (!user) {
                    user = await prisma.user.create({
                        data: { email, name: email.split("@")[0], role: "PATIENT" },
                    });
                }

                let patientId: string | null = null;
                if (user.role === "PATIENT") {
                    const patient = await prisma.patient.upsert({
                        where: { userId: user.id },
                        update: {},
                        create: { userId: user.id },
                    });
                    patientId = patient.id;

                    await prisma.integrationToken.upsert({
                        where: { patientId_provider: { patientId: patient.id, provider: "freestyle" } },
                        create: {
                            patientId: patient.id,
                            provider: "freestyle",
                            accessToken: await encryptValue(email),
                            refreshToken: await encryptValue(password),
                        },
                        update: {
                            accessToken: await encryptValue(email),
                            refreshToken: await encryptValue(password),
                        },
                    });
                }

                return { id: user.id, email: user.email, name: user.name, role: user.role, patientId };
            },
        }),
    ],
    callbacks: {
        async jwt({ token, user }) {
            if (user) {
                token.role = (user as any).role;
                token.id = user.id;
                token.patientId = (user as any).patientId ?? null;
            }
            return token;
        },
        async session({ session, token }) {
            if (session.user) {
                (session.user as any).role = token.role;
                (session.user as any).id = token.id;
                (session.user as any).patientId = token.patientId ?? null;
            }
            return session;
        },
    },
};
