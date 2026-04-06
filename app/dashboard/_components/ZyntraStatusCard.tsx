"use client";

// ─────────────────────────────────────────────────────────────────────────────
// ZyntraStatusCard — Human-First Design
// Hierarchy: Glucose value → Status sentence → Context → Score → Factors → Action
// ─────────────────────────────────────────────────────────────────────────────

export interface ZyntraHackStatus {
  participantId: string;
  scenario: string;
  /** Detected predictive scenario from RiskAgent */
  glucoseScenario: 'low' | 'high' | 'unstable' | 'stable';
  latestSignals: {
    glucose: { value: number; unit: string; timestamp: Date } | null;
    activity: { steps: number; timestamp: Date } | null;
  };
  riskScore: number;
  riskLevel: "low" | "medium" | "high";
  contributingFactors: string[];
  explanation: string;
  primaryMessage: string;
  secondaryMessage: string;
  coachMessage: string;
  voicePayload: { voiceText: string; shouldSpeak: boolean };
  alertPayload: {
    alertTitle: string;
    alertMessage: string;
    alertSeverity: string;
    lockscreenPreview: string;
    voiceText: string;
  };
}

interface ZyntraStatusCardProps {
  output: ZyntraHackStatus;
  loading?: boolean;
  onTalkToZyntra: () => void;
  onSpeak: (text: string) => void;
}

// ─── Color system ────────────────────────────────────────────────────────────
// HIGH     → red-600   (danger, urgent)
// LOW      → orange-600 (hypo warning, different from high)
// UNSTABLE → amber-500  (caution, watch)
// STABLE   → emerald-600 (safe, calm)

function getStateConfig(glucoseScenario: string) {
  if (glucoseScenario === 'high') {
    return {
      accent: 'text-amber-600',
      accentBg: 'bg-amber-50',
      accentBorder: 'border-amber-200',
      glucoseColor: 'text-amber-600',
      dot: 'bg-amber-500',
      badge: 'bg-amber-100 text-amber-700 border-amber-200',
      label: 'Watch',
    };
  }
  if (glucoseScenario === 'low') {
    return {
      accent: 'text-orange-600',
      accentBg: 'bg-orange-50',
      accentBorder: 'border-orange-200',
      glucoseColor: 'text-orange-600',
      dot: 'bg-orange-500',
      badge: 'bg-orange-100 text-orange-700 border-orange-200',
      label: 'Low',
    };
  }
  if (glucoseScenario === 'unstable') {
    return {
      accent: 'text-amber-600',
      accentBg: 'bg-amber-50',
      accentBorder: 'border-amber-200',
      glucoseColor: 'text-amber-600',
      dot: 'bg-amber-400',
      badge: 'bg-amber-100 text-amber-700 border-amber-200',
      label: 'Watch',
    };
  }
  return {
    accent: 'text-emerald-600',
    accentBg: 'bg-emerald-50',
    accentBorder: 'border-emerald-200',
    glucoseColor: 'text-emerald-600',
    dot: 'bg-emerald-500',
    badge: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    label: 'Stable',
  };
}

export function ZyntraStatusCard({
  output,
  loading,
  onTalkToZyntra,
  onSpeak,
}: ZyntraStatusCardProps) {
  const cfg = getStateConfig(output.glucoseScenario ?? output.riskLevel);
  const glucose = output.latestSignals.glucose;

  return (
    <div
      id="zyntra-status-card"
      className="rounded-[2.5rem] bg-white border border-slate-100 shadow-[0_24px_60px_rgba(0,0,0,0.06)] transition-all duration-500 overflow-hidden"
    >
      {/* ── Top colored accent strip ── */}
      <div className={`h-1.5 w-full ${cfg.dot} opacity-60`} />

      <div className="p-8 pb-7">

        {/* ── BLOCK 1 · Glucose value (LARGEST element) ── */}
        <div className="flex items-end justify-between mb-6">
          <div>
            <div className={`text-[72px] font-black leading-none tracking-tighter ${cfg.glucoseColor}`}>
              {glucose?.value ?? "—"}
            </div>
            <div className="text-slate-400 text-sm font-bold uppercase tracking-widest mt-1">
              {glucose?.unit ?? "mg/dL"}
            </div>
          </div>

          {/* Live pulse indicator */}
          <div className="flex flex-col items-end gap-2 mb-1">
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${cfg.dot} animate-pulse`} />
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">Live</span>
            </div>
            {/* Risk badge — small, secondary */}
            <div className={`px-3 py-1 rounded-full border text-[11px] font-black uppercase tracking-wider ${cfg.badge}`}>
              {cfg.label} · {output.riskScore}/100
            </div>
          </div>
        </div>

        {/* ── BLOCK 2 · Primary human sentence ── */}
        <div className="mb-6">
          <h2 className="text-2xl font-black text-slate-900 leading-tight tracking-tight mb-1">
            {output.primaryMessage}
          </h2>
          <p className="text-slate-400 text-[15px] font-medium">
            {output.secondaryMessage}
          </p>
        </div>

        {/* ── BLOCK 3 · Contributing factors (max 3 chips) ── */}
        {output.contributingFactors.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-6">
            {output.contributingFactors.slice(0, 3).map((factor, i) => (
              <span
                key={i}
                className="px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-full text-[11px] font-semibold text-slate-500 lowercase"
              >
                {factor}
              </span>
            ))}
          </div>
        )}

        {/* ── BLOCK 4 · Coach action (with speaker) ── */}
        <div className={`p-5 rounded-[1.75rem] border ${cfg.accentBg} ${cfg.accentBorder} mb-6`}>
          <div className="flex items-center justify-between gap-4">
            <p className={`text-[16px] font-black leading-snug flex-1 ${cfg.accent}`}>
              {output.coachMessage}
            </p>
            <button
              id="zyntra-speak-btn"
              onClick={() => onSpeak(output.voicePayload.voiceText)}
              aria-label="Hear voice guidance"
              className={`w-12 h-12 rounded-full flex items-center justify-center border transition-all active:scale-90 shadow-md flex-shrink-0 ${cfg.accentBg} ${cfg.accentBorder}`}
            >
              <svg className={`w-5 h-5 ${cfg.accent}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2.5}
                  d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* ── BLOCK 5 · Primary CTA ── */}
        <button
          id="zyntra-assistant-btn"
          onClick={onTalkToZyntra}
          disabled={loading}
          style={{ backgroundColor: "#08253F" }}
          className="w-full flex items-center justify-center gap-2 py-4 rounded-[1.25rem] text-white text-sm font-black uppercase tracking-widest shadow-lg hover:opacity-90 transition-all active:scale-[0.98] disabled:opacity-50"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5}
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          Open Zyntra Assistant
        </button>
      </div>
    </div>
  );
}
