import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { prisma } from "@/server/db/prisma";
import { z } from "zod";

const BaselineRequestSchema = z.object({
    patientId: z.string().min(1),
    windowDays: z.number().int().positive().default(14),
});

export async function POST(req: NextRequest) {
    // ── Auth (ADMIN / CLINICIAN only) ──────────────────────────
    const auth = await requireRole("ADMIN", "CLINICIAN");
    if (!auth.authorized) return auth.response;

    let body: unknown;
    try {
        body = await req.json();
    } catch {
        return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
    }

    const parsed = BaselineRequestSchema.safeParse(body);
    if (!parsed.success) {
        return NextResponse.json(
            { error: "Validation failed", details: parsed.error.flatten() },
            { status: 400 }
        );
    }

    const { patientId, windowDays } = parsed.data;

    try {
        const since = new Date(Date.now() - windowDays * 24 * 60 * 60 * 1000);

        // Load all numeric (non-encrypted) signals in the window
        const signals = await prisma.signal.findMany({
            where: {
                patientId,
                ts: { gte: since },
                value: { not: null },
            },
        });

        // Group by type
        const groups = new Map<string, number[]>();
        for (const s of signals) {
            if (s.value === null) continue;
            const arr = groups.get(s.type) ?? [];
            arr.push(s.value);
            groups.set(s.type, arr);
        }

        const upserted = [];

        for (const [metric, values] of groups) {
            if (values.length < 2) continue; // need at least 2 for std

            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const variance =
                values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (values.length - 1);
            const std = Math.sqrt(variance);

            const row = await prisma.baseline.upsert({
                where: {
                    patientId_metric_windowDays: { patientId, metric, windowDays },
                },
                update: { mean, std },
                create: { patientId, metric, windowDays, mean, std },
            });

            upserted.push({ metric, windowDays, mean, std, id: row.id });
        }

        return NextResponse.json({ ok: true, baselines: upserted });
    } catch (err) {
        console.error("Baseline error:", err);
        return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
}
