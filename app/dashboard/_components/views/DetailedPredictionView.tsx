import React from "react";
import { PredictionResult } from "../../_lib/mockRiskData";
import { RiskBadge } from "../ui/RiskBadge";
import { ActionCard } from "../ui/ActionCard";

interface DetailedPredictionViewProps {
    data: PredictionResult;
    timeframe: "48h" | "72h";
    onBack: () => void;
}

export function DetailedPredictionView({ data, timeframe, onBack }: DetailedPredictionViewProps) {
    const score = timeframe === "48h" ? data.score48h : data.score72h;
    const riskLevel = timeframe === "48h" ? data.riskLevel48h : data.riskLevel72h;
    
    // Quick static chart visualization for the drill down
    const trendData = timeframe === "48h" ? [30, 42, 55, 68, score] : [42, 55, 68, 79, score];

    return (
        <div className="pb-8 animate-in slide-in-from-right-4 fade-in duration-300">
            {/* Nav */}
            <button onClick={onBack} className="text-slate-500 mb-6 flex items-center gap-2 hover:text-slate-900 transition-colors font-medium text-sm">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Dashboard
            </button>

            {/* Header */}
            <div className="flex justify-between items-start mb-6">
                <div>
                    <h1 className="text-2xl font-serif font-bold text-slate-900 mb-1">{timeframe} Risk Analysis</h1>
                    <p className="text-slate-500">Predicted deviation from baseline</p>
                </div>
                <RiskBadge level={riskLevel} />
            </div>

            {/* Big Score & Chart */}
            <div className="bg-white rounded-[1.5rem] p-6 shadow-sm border border-slate-100 mb-8 relative overflow-hidden">
                <div className="flex items-end gap-3 mb-6 relative z-10">
                    <span className="text-6xl font-serif font-bold tracking-tighter text-slate-900 leading-none">{score}</span>
                    <span className="text-slate-500 font-medium pb-1">Score / 100</span>
                </div>
                
                <div className="h-32 flex items-end gap-2 mt-4 relative z-10">
                    {trendData.map((val, i) => (
                        <div key={i} className="flex-1 flex flex-col items-center gap-2">
                            <div className="w-full bg-slate-100 rounded-t-md relative flex items-end overflow-hidden" style={{ height: '100px' }}>
                                <div 
                                    className={`w-full rounded-t-md transition-all duration-1000 ease-out delay-${i * 100} ${i === trendData.length - 1 ? 'bg-zyntra-navy' : 'bg-slate-300'}`} 
                                    style={{ height: `${val}%` }} 
                                />
                            </div>
                            <span className="text-[10px] text-slate-400 font-bold">T-{trendData.length - 1 - i}</span>
                        </div>
                    ))}
                </div>
                <div className="absolute top-0 right-0 p-4 opacity-[0.02]">
                    <svg className="w-32 h-32 text-zyntra-navy" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 22h20L12 2z"/></svg>
                </div>
            </div>

            {/* Prediction meaning */}
            <div className="mb-10">
                <h3 className="text-lg font-serif font-bold text-slate-900 mb-2">What happens next</h3>
                <p className="text-slate-600 leading-relaxed text-[15px]">{data.predictedEvent}. The projected trend indicates a shift in your metabolic stability over the chosen window. This does not diagnose a clinical emergency.</p>
            </div>

            {/* Contributor breakdown */}
            <div className="mb-10">
                <h3 className="text-lg font-serif font-bold text-slate-900 mb-4">Contributor Breakdown</h3>
                <div className="space-y-3">
                    {data.drivers.map((driver) => (
                        <div key={driver.key} className="flex items-start gap-3 bg-white p-4 rounded-2xl border border-slate-100 shadow-sm">
                             <div className={`mt-1 flex-shrink-0 w-2 h-2 rounded-full ${driver.impact === 'high' ? 'bg-rose-500' : driver.impact === 'medium' ? 'bg-amber-400' : 'bg-teal-400'}`} />
                             <div>
                                 <h4 className="font-bold text-slate-900 text-sm">{driver.label}</h4>
                                 <p className="text-slate-500 text-xs mt-1">{driver.explanation}</p>
                             </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Action Support */}
            <div>
                <h3 className="text-lg font-serif font-bold text-slate-900 mb-4">Supportive Actions</h3>
                <div className="space-y-3">
                    {data.recommendations.slice(0, 2).map((rec) => (
                        <ActionCard 
                            key={rec.id}
                            title={rec.title}
                            description={rec.description}
                            icon={<svg className="w-5 h-5 text-zyntra-navy" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>}
                        />
                    ))}
                </div>
            </div>

        </div>
    );
}
