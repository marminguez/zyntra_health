import { getServerSession } from "next-auth";
import { NextResponse } from "next/server";
import { authOptions } from "./auth";

type Role = "ADMIN" | "CLINICIAN" | "PATIENT" | "SERVICE";

/**
 * Checks that the current session user has one of the allowed roles.
 * Returns the session if authorised, or a 401/403 NextResponse otherwise.
 */
export async function requireRole(...allowed: Role[]) {
    const session = await getServerSession(authOptions);

    if (!session?.user) {
        return {
            authorized: false as const,
            response: NextResponse.json({ error: "Unauthenticated" }, { status: 401 }),
        };
    }

    const role = (session.user as any).role as Role | undefined;

    if (!role || !allowed.includes(role)) {
        return {
            authorized: false as const,
            response: NextResponse.json({ error: "Forbidden" }, { status: 403 }),
        };
    }

    return {
        authorized: true as const,
        session,
        userId: (session.user as any).id as string,
        role,
    };
}

/**
 * Requires any authenticated user (all roles).
 */
export async function requireAuth() {
    return requireRole("ADMIN", "CLINICIAN", "PATIENT", "SERVICE");
}
