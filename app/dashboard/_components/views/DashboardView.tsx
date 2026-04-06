import React from "react";
import { PredictionResult, Driver, Recommendation } from "../../_lib/mockRiskData";
import { RiskBadge } from "../ui/RiskBadge";
import { ActionCard } from "../ui/ActionCard";
import { useSession } from "next-auth/react";

interface DashboardViewProps {
    data: PredictionResult;
    onDrilldown: (timeframe: "48h" | "72h") => void;
}

export function DashboardView({ data, onDrilldown }: DashboardViewProps) {
    const { data: session } = useSession();
    const userName = (session?.user as any)?.email?.split("@")[0] || "User";

    let escalationMessage = "Monitor closely if this risk remains high for more than 2 days.";
    if (data.escalationFlag === "human_review") {
        escalationMessage = "Repeated highs or lows may require human review. Consider discussing this pattern with your clinician if instability persists.";
    }

    return (
        <div className="pb-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Header / Welcome */}
            <div className="mb-8">
                <p className="text-slate-500 font-medium mb-1 tracking-wide">Good morning, {userName}</p>
                <h1 className="text-3xl font-serif font-bold text-slate-900 mb-2">{data.summary}</h1>
                <p className="text-slate-600 text-[15px] leading-relaxed border-l-2 border-zyntra-teal pl-3 mt-4">
                    {data.predictedEvent}
                </p>
            </div>

            {/* Main Prediction Cards */}
            <div className="flex gap-4 mb-8">
                <div 
                    onClick={() => onDrilldown("48h")}
                    className="flex-1 bg-gradient-to-br from-white to-slate-50 rounded-[1.5rem] p-5 shadow-sm border border-slate-100 cursor-pointer hover:shadow-md transition-all relative overflow-hidden group"
                >
                    <div className="absolute -top-4 -right-4 p-4 opacity-[0.03] group-hover:scale-110 group-hover:opacity-10 transition-all duration-300">
                        <svg className="w-24 h-24 text-zyntra-navy" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 22h20L12 2z"/></svg>
                    </div>
                    <h3 className="text-slate-500 font-bold text-xs uppercase tracking-widest mb-3 relative z-10">48H Outlook</h3>
                    <div className="flex items-end gap-3 mb-3 relative z-10">
                        <span className="text-5xl font-serif font-bold tracking-tighter text-slate-900 leading-none">{data.score48h}</span>
                        <div className="pb-1"><RiskBadge level={data.riskLevel48h} /></div>
                    </div>
                </div>

                <div 
                    onClick={() => onDrilldown("72h")}
                    className="flex-1 bg-gradient-to-br from-white to-slate-50 rounded-[1.5rem] p-5 shadow-sm border border-slate-100 cursor-pointer hover:shadow-md transition-all relative overflow-hidden group"
                >
                    <div className="absolute -top-4 -right-4 p-4 opacity-[0.03] group-hover:scale-110 group-hover:opacity-10 transition-all duration-300">
                        <svg className="w-24 h-24 text-zyntra-teal" fill="currentColor" viewBox="0 0 24 24"><path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-11v6h2v-6h-2zm0-4v2h2V7h-2z"/></svg>
                    </div>
                    <h3 className="text-slate-500 font-bold text-xs uppercase tracking-widest mb-3 relative z-10">72H Trajectory</h3>
                    <div className="flex items-end gap-3 mb-3 relative z-10">
                        <span className="text-5xl font-serif font-bold tracking-tighter text-slate-900 leading-none">{data.score72h}</span>
                        <div className="pb-1"><RiskBadge level={data.riskLevel72h} /></div>
                    </div>
                </div>
            </div>

            {/* What this means */}
            <div className="bg-slate-100 rounded-[1.25rem] p-5 mb-10 text-slate-600 text-sm leading-relaxed border border-slate-200/60">
                <span className="font-bold text-slate-900 block mb-1">What this means</span>
                This score estimates the likelihood that your metabolic stability may worsen relative to your usual baseline in the next 48–72 hours. It does not predict an exact glucose value and does not replace medical judgement.
            </div>

            {/* Top Drivers */}
            <div className="mb-10">
                <h2 className="text-xl font-serif font-bold text-slate-900 mb-4">Top Drivers</h2>
                <div className="space-y-3">
                    {data.drivers.map((driver) => (
                        <ActionCard 
                            key={driver.key}
                            title={driver.label}
                            description={driver.explanation}
                            icon={
                                driver.direction === "up" 
                                    ? <svg className="w-5 h-5 text-rose-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" /></svg>
                                    : <svg className="w-5 h-5 text-teal-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
                            }
                        />
                    ))}
                </div>
            </div>

            {/* Recommendations */}
            <div className="mb-10">
                <h2 className="text-xl font-serif font-bold text-slate-900 mb-4">What you can do today</h2>
                <div className="space-y-3">
                    {data.recommendations.map((rec) => (
                        <ActionCard 
                            key={rec.id}
                            title={rec.title}
                            description={rec.description}
                            icon={<svg className="w-5 h-5 text-zyntra-navy" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>}
                        />
                    ))}
                </div>
            </div>

            {/* Escalation */}
            {data.escalationFlag !== "none" && (
                <div className={`rounded-[1.25rem] p-5 border shadow-sm ${data.escalationFlag === "human_review" ? "bg-rose-50 border-rose-100" : "bg-amber-50 border-amber-100"}`}>
                    <h3 className={`font-bold mb-2 flex items-center gap-2 ${data.escalationFlag === "human_review" ? "text-rose-900" : "text-amber-900"}`}>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        When to pay closer attention
                    </h3>
                    <p className={`text-sm leading-relaxed ${data.escalationFlag === "human_review" ? "text-rose-800" : "text-amber-800"}`}>
                        {escalationMessage}
                    </p>
                </div>
            )}
        </div>
    );
}
