import { NextRequest, NextResponse } from "next/server";
import { requireAuth } from "@/server/auth/rbac";
import { checkRateLimit } from "@/server/security/rateLimit";
import { SignalIngestSchema } from "@/server/zyntra/schemas";
import { ingestSignal, ConsentError } from "@/server/zyntra/ingest";

export async function POST(req: NextRequest) {
    // ── Rate limit ─────────────────────────────────────────────
    const ip = req.headers.get("x-forwarded-for") ?? req.ip ?? "unknown";
    if (!checkRateLimit(ip, "ingest")) {
        return NextResponse.json({ error: "Rate limit exceeded" }, { status: 429 });
    }

    // ── Auth ───────────────────────────────────────────────────
    const auth = await requireAuth();
    if (!auth.authorized) return auth.response;

    // ── Validate ───────────────────────────────────────────────
    let body: unknown;
    try {
        body = await req.json();
    } catch {
        return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
    }

    const parsed = SignalIngestSchema.safeParse(body);
    if (!parsed.success) {
        return NextResponse.json(
            { error: "Validation failed", details: parsed.error.flatten() },
            { status: 400 }
        );
    }

    // ── Ingest ─────────────────────────────────────────────────
    try {
        const result = await ingestSignal(
            parsed.data,
            auth.userId,
            ip,
            req.headers.get("user-agent")
        );
        return NextResponse.json({ ok: true, signalId: result.id }, { status: 201 });
    } catch (err) {
        if (err instanceof ConsentError) {
            return NextResponse.json({ error: err.message }, { status: 403 });
        }
        console.error("Ingest error:", err);
        return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
}
