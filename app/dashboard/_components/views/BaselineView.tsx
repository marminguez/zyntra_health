"use client";
import React from "react";

type Scenario = "low" | "high" | "unstable" | "stable";

interface Props {
  glucoseScenario: Scenario;
}

// ── Scenario-specific content ────────────────────────────────────────────────

const SCENARIO_CONTENT: Record<Scenario, {
  banner: { bg: string; border: string; text: string; label: string; body: string };
  glucoseStatus: { bar: string; width: string; label: string; delta: string; deltaColor: string };
  activityStatus: { bar: string; width: string; label: string; delta: string; deltaColor: string };
  sleepStatus: { bar: string; width: string; label: string; delta: string; deltaColor: string };
  tirLabel: string;
  tirBars: number[];
  tirNote: string;
}> = {
  low: {
    banner: {
      bg: "bg-orange-50", border: "border-orange-100", text: "text-orange-900",
      label: "⚠️ Low glucose pattern detected",
      body: "Zyntra is comparing today's signals against your personal baseline. Your glucose is tracking below your usual floor — this pattern indicates you may need to eat soon.",
    },
    glucoseStatus: { bar: "bg-orange-400", width: "35%", label: "7h 15m Avg", delta: "−29 mg/dL vs. baseline", deltaColor: "text-orange-600" },
    activityStatus: { bar: "bg-slate-300", width: "50%", label: "4,800 Steps Today", delta: "Below baseline", deltaColor: "text-slate-400" },
    sleepStatus: { bar: "bg-rose-300", width: "60%", label: "6h 10m Last Night", delta: "−1h 5m", deltaColor: "text-rose-500" },
    tirLabel: "58%",
    tirBars: [78, 72, 65, 58, 50, 42, 38],
    tirNote: "Time-in-range has been declining over the past 7 days — you've been spending more time below the safe floor.",
  },
  high: {
    banner: {
      bg: "bg-amber-50", border: "border-amber-100", text: "text-amber-900",
      label: "📈 Rising glucose trend detected",
      body: "Your glucose baseline is shifting upward. The data shows a pattern of post-meal spikes that are taking longer to resolve compared to your personal average.",
    },
    glucoseStatus: { bar: "bg-amber-400", width: "80%", label: "7h 15m Avg", delta: "+24 mg/dL vs. baseline", deltaColor: "text-amber-600" },
    activityStatus: { bar: "bg-slate-300", width: "45%", label: "4,200 Steps Today", delta: "−34% vs. baseline", deltaColor: "text-slate-500" },
    sleepStatus: { bar: "bg-slate-300", width: "72%", label: "7h 05m Last Night", delta: "Within range", deltaColor: "text-slate-400" },
    tirLabel: "71%",
    tirBars: [85, 80, 77, 74, 72, 68, 71],
    tirNote: "Time-in-range is slightly below your personal average. Post-meal excursions are the primary driver.",
  },
  unstable: {
    banner: {
      bg: "bg-amber-50", border: "border-amber-100", text: "text-amber-900",
      label: "↕️ Variable glucose pattern",
      body: "Your glucose is swinging more than usual without reaching extreme values. Irregular meal timing and sleep disruptions are the most likely contributing factors.",
    },
    glucoseStatus: { bar: "bg-amber-300", width: "60%", label: "7h 15m Avg", delta: "High variability today", deltaColor: "text-amber-600" },
    activityStatus: { bar: "bg-teal-300", width: "70%", label: "6,100 Steps Today", delta: "+5% vs. baseline", deltaColor: "text-teal-600" },
    sleepStatus: { bar: "bg-rose-300", width: "55%", label: "6h 30m Last Night", delta: "−45m", deltaColor: "text-rose-500" },
    tirLabel: "79%",
    tirBars: [85, 78, 83, 70, 75, 80, 79],
    tirNote: "Time-in-range is within acceptable limits, but the variability pattern is notable.",
  },
  stable: {
    banner: {
      bg: "bg-emerald-50", border: "border-emerald-100", text: "text-emerald-900",
      label: "✅ Baseline match — you're on track",
      body: "Your current signals are closely aligned with your personal metabolic baseline. Sleep, activity, and glucose are all within your normal ranges.",
    },
    glucoseStatus: { bar: "bg-emerald-400", width: "88%", label: "7h 15m Avg", delta: "Within baseline", deltaColor: "text-emerald-600" },
    activityStatus: { bar: "bg-teal-400", width: "85%", label: "6,400 Steps Today", delta: "+12% vs. baseline", deltaColor: "text-teal-600" },
    sleepStatus: { bar: "bg-slate-400", width: "75%", label: "7h 20m Last Night", delta: "Stable", deltaColor: "text-slate-400" },
    tirLabel: "88%",
    tirBars: [70, 80, 88, 85, 90, 88, 88],
    tirNote: "Your 14-day time-in-range is 88% — well within your personal target range.",
  },
};

export function BaselineView({ glucoseScenario }: Props) {
  const s = SCENARIO_CONTENT[glucoseScenario];

  return (
    <div className="pb-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-serif font-bold text-slate-900 mb-2">Your Baseline</h1>
        <p className="text-slate-600 text-[15px] leading-relaxed">
          Zyntra learns what is normal for <strong>you</strong>. We compare your current signals against your own trailing patterns.
        </p>
      </div>

      {/* Scenario banner */}
      <div className={`${s.banner.bg} border ${s.banner.border} rounded-[1.5rem] p-5 mb-8`}>
        <p className={`font-bold text-sm ${s.banner.text} mb-1`}>{s.banner.label}</p>
        <p className={`text-sm leading-relaxed ${s.banner.text} opacity-80`}>{s.banner.body}</p>
      </div>

      <h2 className="text-xl font-serif font-bold text-slate-900 mb-4">Metabolic Foundation</h2>

      <div className="space-y-4 mb-8">
        {/* Glucose */}
        <div className="bg-white rounded-[1.5rem] p-5 shadow-sm border border-slate-100">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-indigo-50 rounded-full flex items-center justify-center text-indigo-500">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
              </div>
              <div>
                <h3 className="font-bold text-slate-900">Glucose vs. Baseline</h3>
                <p className="text-xs font-bold text-slate-400 tracking-wider uppercase">{s.glucoseStatus.label}</p>
              </div>
            </div>
            <span className={`text-sm font-bold ${s.glucoseStatus.deltaColor}`}>{s.glucoseStatus.delta}</span>
          </div>
          <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className={`h-full ${s.glucoseStatus.bar} rounded-full transition-all duration-700`} style={{ width: s.glucoseStatus.width }} />
          </div>
        </div>

        {/* Activity */}
        <div className="bg-white rounded-[1.5rem] p-5 shadow-sm border border-slate-100">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-teal-50 rounded-full flex items-center justify-center text-teal-500">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              </div>
              <div>
                <h3 className="font-bold text-slate-900">Activity</h3>
                <p className="text-xs font-bold text-slate-400 tracking-wider uppercase">{s.activityStatus.label}</p>
              </div>
            </div>
            <span className={`text-sm font-bold ${s.activityStatus.deltaColor}`}>{s.activityStatus.delta}</span>
          </div>
          <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className={`h-full ${s.activityStatus.bar} rounded-full transition-all duration-700`} style={{ width: s.activityStatus.width }} />
          </div>
        </div>

        {/* Sleep */}
        <div className="bg-white rounded-[1.5rem] p-5 shadow-sm border border-slate-100">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-rose-50 rounded-full flex items-center justify-center text-rose-500">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>
              </div>
              <div>
                <h3 className="font-bold text-slate-900">Sleep</h3>
                <p className="text-xs font-bold text-slate-400 tracking-wider uppercase">{s.sleepStatus.label}</p>
              </div>
            </div>
            <span className={`text-sm font-bold ${s.sleepStatus.deltaColor}`}>{s.sleepStatus.delta}</span>
          </div>
          <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className={`h-full ${s.sleepStatus.bar} rounded-full transition-all duration-700`} style={{ width: s.sleepStatus.width }} />
          </div>
        </div>
      </div>

      {/* TIR card */}
      <div className="bg-zyntra-navy text-white rounded-[1.5rem] p-6 shadow-md relative overflow-hidden">
        <div className="relative z-10">
          <div className="flex justify-between items-start mb-1">
            <h3 className="font-serif font-bold text-xl">Glucose Stability</h3>
            <span className="text-white font-black text-2xl">{s.tirLabel}</span>
          </div>
          <p className="text-slate-400 text-sm mb-4 leading-relaxed">{s.tirNote}</p>
          <div className="flex items-end gap-2 h-20 mt-8">
            {s.tirBars.map((h, i) => (
              <div key={i} className="flex-1 bg-zyntra-teal/20 rounded-t-sm relative">
                <div className="absolute bottom-0 w-full bg-zyntra-teal rounded-t-sm transition-all duration-700" style={{ height: `${h}%` }} />
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500 mt-3 uppercase tracking-widest text-right">7-day time-in-range</p>
        </div>
      </div>
    </div>
  );
}
