"use client";
import React from "react";

type Scenario = "low" | "high" | "unstable" | "stable";

interface Props {
  glucoseScenario: Scenario;
}

const SCENARIO_HISTORY: Record<Scenario, {
  chartTitle: string;
  chartData: number[]; // 7 values for area chart (height %)
  drivers: Array<{ count: string; color: string; title: string; description: string; icon: React.ReactNode }>;
  reminderBody: string;
}> = {
  low: {
    chartTitle: "Low Pattern Frequency (Last 7 Days)",
    chartData: [15, 25, 20, 45, 30, 70, 85], // Rising recently
    drivers: [
      {
        count: "4x", color: "text-orange-500 bg-orange-50", title: "Skipped or Delayed Meals",
        description: "In the last two weeks, pushing your meals out by more than 5 hours correlated strongly with sudden below-baseline drops.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
      },
      {
        count: "2x", color: "text-amber-500 bg-amber-50", title: "High Output Exercise on Empty",
        description: "Intense runs before breakfast preceded mid-morning instability and low floors twice this week.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
      }
    ],
    reminderBody: "Recurring lows often mean your body is exhausting its readily available energy sooner than expected. Prioritize consistent meal timing to break this trend."
  },
  high: {
    chartTitle: "Spike Pattern Frequency (Last 7 Days)",
    chartData: [20, 15, 10, 40, 60, 50, 75], // Recent spikes
    drivers: [
      {
        count: "3x", color: "text-rose-500 bg-rose-50", title: "Late Heavy Meals + Poor Sleep",
        description: "Eating dense carbohydrates after 8 PM strongly predicted poor sleep architecture and elevated morning glucose the next day.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>
      },
      {
        count: "2x", color: "text-teal-600 bg-teal-50", title: "Lower Activity Load",
        description: "Days with under 4,000 steps frequently preceded a jump in your 48h rising risk score and slower meal clearance.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m12-5v8m-12 0h14a2 2 0 002-2V8a2 2 0 00-2-2H6a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
      }
    ],
    reminderBody: "Spikes are normal, but if they take hours to come down, your system is working overtime. Focus on a brief walk after your largest meal to help clear the glucose."
  },
  unstable: {
    chartTitle: "Variability Score (Last 7 Days)",
    chartData: [30, 25, 40, 35, 60, 80, 70], // High variability
    drivers: [
      {
        count: "5x", color: "text-indigo-500 bg-indigo-50", title: "Inconsistent Rhythms",
        description: "Varying your wake time and first meal by more than 2 hours across the week has driven higher overall glucose volatility.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
      },
      {
        count: "3x", color: "text-rose-500 bg-rose-50", title: "High Work/Stress Days",
        description: "Self-reported 'high stress' days correlate with jagged glucose lines, even when meals are generally healthy.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
      }
    ],
    reminderBody: "Variability is often a sign of circadian mismatch or stress. Try to anchor your day with fixed times for waking up, eating, and winding down."
  },
  stable: {
    chartTitle: "Stability Maintenance (Last 7 Days)",
    chartData: [85, 90, 88, 85, 92, 90, 95], // High and flat
    drivers: [
      {
        count: "7x+", color: "text-emerald-500 bg-emerald-50", title: "Consistent Sleep Foundation",
        description: "You've successfully maintained 7+ hours of quality sleep exactly matching your biological rhythm. This is deeply stabilizing your baseline.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>
      },
      {
        count: "4x", color: "text-teal-600 bg-teal-50", title: "Daily Movement",
        description: "Hitting your ~6,000 step average ensures your muscles remain sensitive to insulin, preventing spikes.",
        icon: <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m12-5v8m-12 0h14a2 2 0 002-2V8a2 2 0 00-2-2H6a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
      }
    ],
    reminderBody: "Your body thrives on this routine. You don't need to over-optimize; simple, consistent habits are your most powerful tool."
  }
};

export function HistoryView({ glucoseScenario }: Props) {
  const content = SCENARIO_HISTORY[glucoseScenario];

  return (
    <div className="pb-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-serif font-bold text-slate-900 mb-2">Trends & Patterns</h1>
        <p className="text-slate-600 text-[15px] leading-relaxed">
          Understanding your trailing risk history. Focus on the underlying habits driving your {glucoseScenario.toUpperCase()} patterns.
        </p>
      </div>

      <div className="bg-white rounded-[1.5rem] p-6 shadow-sm border border-slate-100 mb-8">
        <div className="flex justify-between items-end mb-6">
          <div>
            <h3 className="font-serif font-bold text-lg text-slate-900">{content.chartTitle}</h3>
            <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest">7-Day Trajectory</p>
          </div>
        </div>
        
        {/* Mock Area Chart */}
        <div className="h-40 flex items-end justify-between gap-1.5 mt-4 relative">
          <div className="absolute top-1/2 left-0 w-full border-t border-dashed border-slate-200 z-0"></div>
          {content.chartData.map((val, i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-2 group relative z-10 w-full h-full justify-end">
              <div 
                className="w-full bg-zyntra-teal rounded-t-sm transition-all duration-700 cursor-pointer" 
                style={{ height: `${val}%`, minHeight: '4px' }} 
              />
              <span className={`text-[10px] pb-1 font-bold ${-6 + i === 0 ? 'text-zyntra-navy' : 'text-slate-400'}`}>
                {-6 + i === 0 ? 'Today' : ((-6 + i) + 'd')}
              </span>
            </div>
          ))}
        </div>
      </div>

      <h2 className="text-xl font-serif font-bold text-slate-900 mb-4">Leading Drivers</h2>
      <div className="space-y-4 mb-8">
        {content.drivers.map((driver, idx) => (
          <div key={idx} className="bg-white p-5 rounded-[1.25rem] border border-slate-100 shadow-sm flex items-start gap-4">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 ${driver.color}`}>
              {driver.icon ? driver.icon : <span className="font-bold text-lg">{driver.count}</span>}
            </div>
            <div>
              <h4 className="font-bold text-slate-900">{driver.title}</h4>
              <p className="text-sm text-slate-600 mt-1 leading-relaxed">{driver.description}</p>
            </div>
          </div>
        ))}
      </div>
      
      <div className="bg-amber-50 border border-amber-100 rounded-[1.25rem] p-5 shadow-sm text-amber-900">
        <h3 className="font-bold mb-1 flex items-center gap-2">
          <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          Gentle Reminder
        </h3>
        <p className="text-sm leading-relaxed text-amber-800">
          {content.reminderBody}
        </p>
      </div>
    </div>
  );
}
