import { NextRequest, NextResponse } from "next/server";
import { requireRole } from "@/server/auth/rbac";
import { checkRateLimit } from "@/server/security/rateLimit";
import { RiskRequestSchema } from "@/server/zyntra/schemas";
import { prisma } from "@/server/db/prisma";
import { decryptNumber } from "@/server/security/crypto";
import { computeRisk, computeRiskML, VALID_HORIZONS, HorizonHrs, ZScores } from "@/server/zyntra/engine";
import { writeAudit } from "@/server/security/audit";
import { loadModels, computePCI, predictTarget } from "@/server/zyntra/inference";
import { resolvePatientIdForUser } from "@/server/auth/patientAccess";

/** Map signal type → z-score domain key. */
const SIGNAL_TO_DOMAIN: Record<string, keyof ZScores> = {
    cgm_glucose_mgdl: "glucose",
    hr_bpm: "activity",
    hrv_ms: "hrv",
    sleep_hours: "sleep",
    steps: "activity",
};

// Ensure models are loaded once.
let modelsLoaded = false;
try {
    modelsLoaded = loadModels();
} catch (e) {
    console.error("Failed to load models at startup", e);
}

export async function POST(req: NextRequest) {
    // ── Rate limit ─────────────────────────────────────────────
    const ip = req.headers.get("x-forwarded-for") ?? req.ip ?? "unknown";
    if (!checkRateLimit(ip, "risk")) {
        return NextResponse.json({ error: "Rate limit exceeded" }, { status: 429 });
    }

    // ── Auth (ADMIN / CLINICIAN / PATIENT) ─────────────────────
    const auth = await requireRole("ADMIN", "CLINICIAN", "PATIENT");
    if (!auth.authorized) return auth.response;

    // ── Validate ───────────────────────────────────────────────
    let body: unknown;
    try {
        body = await req.json();
    } catch {
        return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
    }

    const parsed = RiskRequestSchema.safeParse(body);
    if (!parsed.success) {
        return NextResponse.json(
            { error: "Validation failed", details: parsed.error.flatten() },
            { status: 400 }
        );
    }

    const patientId = await resolvePatientIdForUser(
        auth.role,
        auth.userId,
        parsed.data.patientId
    );

    try {
        // ── Load baselines (windowDays=14) ─────────────────────
        const baselines = await prisma.baseline.findMany({
            where: { patientId, windowDays: 14 },
        });

        const baselineMap = new Map(baselines.map((b) => [`${b.metric}:${b.windowDays}`, b]));

        // ── Load latest 24h signals ────────────────────────────
        const since = new Date(Date.now() - 24 * 60 * 60 * 1000);
        const signals = await prisma.signal.findMany({
            where: { patientId, ts: { gte: since } },
            orderBy: { ts: "desc" },
        });

        // Resolve latest value per type (decrypt if needed)
        const latestByType = new Map<string, number>();
        for (const sig of signals) {
            if (latestByType.has(sig.type)) continue; // already have latest
            let val: number | null = sig.value;
            if (val === null && sig.encryptedValue) {
                val = await decryptNumber(sig.encryptedValue);
            }
            if (val !== null) {
                latestByType.set(sig.type, val);
            }
        }

        // ── ML Feature Engineering ─────────────────────────────
        const getLatest = (type: string) => latestByType.get(type) ?? 0;
        const getBaseline = (type: string) => baselineMap.get(`${type}:14`)?.mean ?? getLatest(type);
        const delta = (curr: number, base: number) => base === 0 ? 0 : (curr - base) / Math.max(1e-6, Math.abs(base));

        const d_hrv = delta(getLatest("hrv_ms"), getBaseline("hrv_ms"));
        const d_sleep = delta(getLatest("sleep_hours"), getBaseline("sleep_hours"));
        const d_stress = delta(getLatest("stress_score"), getBaseline("stress_score"));
        const d_rhr = delta(getLatest("resting_hr_bpm"), getBaseline("resting_hr_bpm"));
        const d_temp = delta(getLatest("body_temp_c"), getBaseline("body_temp_c"));

        const d_g_cv = delta(getLatest("glucose_cv"), getBaseline("glucose_cv"));
        const d_tar = delta(getLatest("TAR"), getBaseline("TAR"));
        const d_tbr = delta(getLatest("TBR"), getBaseline("TBR"));
        const g_trend = getLatest("glucose_trend_7d_slope");

        const pci = computePCI(d_hrv, d_sleep, d_stress, d_rhr, d_temp);

        const features = {
            "delta_glucose_cv": d_g_cv,
            "delta_TAR": d_tar,
            "delta_TBR": d_tbr,
            "pci": pci,
            "glucose_trend_7d": g_trend,
            "delta_sleep": d_sleep,
            "delta_stress": d_stress
        };

        // ── Inference ──────────────────────────────────────────
        let usedModel: "ml" | "zscore" = "zscore";
        let payload: any = {};
        const results = [];

        // Try load models if not loaded yet
        if (!modelsLoaded) modelsLoaded = loadModels();

        if (modelsLoaded) {
            usedModel = "ml";

            const p_hyper_24 = predictTarget("y_hyper_24", features);
            const p_hyper_72 = predictTarget("y_hyper_72", features);
            const p_hypo_12 = predictTarget("y_hypo_12", features);
            const p_hypo_24 = predictTarget("y_hypo_24", features);
            const p_tir_drop_24 = predictTarget("y_tir_drop_24", features);
            const p_tir_drop_72 = predictTarget("y_tir_drop_72", features);
            const p_instability_24 = predictTarget("y_instability_24", features);
            const p_instability_72 = predictTarget("y_instability_72", features);

            const prob_hyper_24 = p_hyper_24?.probability ?? 0;
            const prob_hypo_12 = p_hypo_12?.probability ?? 0;
            const prob_tir_24 = p_tir_drop_24?.probability ?? 0;
            const prob_inst_24 = p_instability_24?.probability ?? 0;
            const prob_hyper_72 = p_hyper_72?.probability ?? 0;
            const prob_hypo_24 = p_hypo_24?.probability ?? 0;
            const prob_tir_72 = p_tir_drop_72?.probability ?? 0;
            const prob_inst_72 = p_instability_72?.probability ?? 0;

            const global_24 = 0.5 * prob_inst_24 + 0.25 * prob_hyper_24 + 0.15 * prob_hypo_12 + 0.10 * prob_tir_24;
            const global_72 = 0.5 * prob_inst_72 + 0.25 * prob_hyper_72 + 0.15 * prob_hypo_24 + 0.10 * prob_tir_72;

            // Guardrails
            const spo2 = getLatest("spo2_pct");
            const temp = getLatest("body_temp_c");
            const ekgFlag = getLatest("ekg_flag");

            const needs_human_review = global_24 > 0.8 || (spo2 > 0 && spo2 < 90) || temp >= 38.0 || ekgFlag > 0;

            payload = {
                usedModel,
                pci,
                global_24,
                global_72,
                p_hyper_24: prob_hyper_24,
                p_hyper_72: prob_hyper_72,
                p_hypo_12: prob_hypo_12,
                p_hypo_24: prob_hypo_24,
                p_tir_drop_24: prob_tir_24,
                p_tir_drop_72: prob_tir_72,
                p_instability_24: prob_inst_24,
                p_instability_72: prob_inst_72,
                needs_human_review,
                warning: needs_human_review ? "Not diagnostic. This triggers human review only." : undefined,
                drivers: {
                    hyper_24: p_hyper_24?.top_drivers ?? [],
                    hypo_12: p_hypo_12?.top_drivers ?? [],
                    instability_24: p_instability_24?.top_drivers ?? []
                }
            };

            // Maintain DB integrity for the /reports view
            for (const h of [24, 72]) {
                const score = h === 24 ? global_24 : global_72;
                const row = await prisma.riskScore.create({
                    data: {
                        patientId,
                        horizonHrs: h,
                        score: score,
                        level: score > 0.66 ? "high" : score > 0.33 ? "medium" : "low",
                        explanation: `ML predictions. Drivers: ${p_instability_24?.top_drivers.map(d => d.feature).join(", ")}`,
                    },
                });
                results.push({ horizonHrs: h, id: row.id });
            }
        } else {
            // ── Fallback Build z-scores ─────────────────────────────────────
            const zScores: ZScores = {};

            for (const [signalType, domain] of Object.entries(SIGNAL_TO_DOMAIN)) {
                const val = latestByType.get(signalType);
                const bl = baselineMap.get(`${signalType}:14`);
                if (val !== undefined && bl && bl.std > 0) {
                    const z = (val - bl.mean) / bl.std;
                    if (zScores[domain] === undefined || Math.abs(z) > Math.abs(zScores[domain]!)) {
                        zScores[domain] = z;
                    }
                }
            }

            const horizons = [24, 48, 72] as const;
            for (const h of horizons) {
                const mlResult = computeRiskML(zScores, h as HorizonHrs);
                const row = await prisma.riskScore.create({
                    data: {
                        patientId,
                        horizonHrs: h,
                        score: mlResult.score,
                        level: mlResult.level,
                        confidence:  mlResult.confidence,
                        explanation: mlResult.fallback
                          ? mlResult.fallbackResult?.explanation ?? "Rule-based fallback"
                          : `ML model v${mlResult.modelVersion} — features: ${mlResult.featuresUsed.join(", ")} (horizon ${h}h)`,
                    },
                });
                results.push({ horizonHrs: h, score: mlResult.score, level: mlResult.level, id: row.id, explanation: row.explanation, confidence: mlResult.confidence });
            }
            payload = { usedModel, results };
        }

        // ── Audit ──────────────────────────────────────────────
        await writeAudit({
            userId: auth.userId,
            action: "RISK_COMPUTED",
            targetId: patientId,
            meta: { horizons: [24, 48, 72], payloadLength: Object.keys(payload).length },
            ip,
            ua: req.headers.get("user-agent"),
        });

        return NextResponse.json({ ok: true, riskScores: results, ...payload });
    } catch (err) {
        console.error("Risk error:", err);
        return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
}
