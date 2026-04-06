"use client";
import React from "react";
import { ActionCard } from "../ui/ActionCard";

type Scenario = "low" | "high" | "unstable" | "stable";

interface Props {
  glucoseScenario: Scenario;
}

const SCENARIO_ACTIONS: Record<Scenario, {
  why: string;
  actions: Array<{ id: string; title: string; description: string; }>;
  monitoring: { title: string; description: string; };
}> = {
  low: {
    why: "We've noticed a pattern of unexpected drops today, bringing your glucose closer to the lower boundary of your baseline. These gentle actions focus on replenishing energy and stabilization.",
    actions: [
      { id: "low-1", title: "Eat a balanced snack soon", description: "Consider something with complex carbohydrates and protein (like an apple with peanut butter) to gently lift and stabilize your level." },
      { id: "low-2", title: "Pause intense physical activity", description: "Vigorous exercise right now could accelerate the drop. Wait until your levels are clearly trending upward." }
    ],
    monitoring: {
      title: "Check back in 20 minutes",
      description: "Notice how you feel shortly after eating. If you use a CGM, ensure the trend arrow flattens or points up before resuming normal activities."
    }
  },
  high: {
    why: "Your current signals show an upward shift. To help your body clear post-meal spikes naturally, these micro-actions target mild activity and hydration.",
    actions: [
      { id: "high-1", title: "Take a 15-minute walk", description: "Light to moderate walking after a meal is one of the most effective non-medical ways to help your muscles use excess glucose." },
      { id: "high-2", title: "Drink a large glass of water", description: "Hydration helps your kidneys flush out excess glucose. Avoid sugary drinks or high-carb snacks right now." }
    ],
    monitoring: {
      title: "Observe post-meal recovery",
      description: "Take note of your glucose 2 hours after your last meal. If your levels remain elevated for long periods, consider adjusting the carb load of your next meal."
    }
  },
  unstable: {
    why: "Your pattern today is highly variable. Large swings up and down can impact your energy and mood. These actions focus on restoring predictability.",
    actions: [
      { id: "unstable-1", title: "Prioritize protein and fiber", description: "For your next meal, ensure it is anchored with protein, fiber, and healthy fats instead of bare carbohydrates to prevent sharp spikes." },
      { id: "unstable-2", title: "Aim for consistent sleep tonight", description: "Disrupted sleep strongly correlates with next-day insulin resistance. Getting quality rest will help reset your baseline." }
    ],
    monitoring: {
      title: "Focus on overall rhythm",
      description: "Pay attention to your meal timing and energy levels. Variability often resolves when daily rhythms (eating, sleeping, moving) are consistent."
    }
  },
  stable: {
    why: "Everything looks aligned. Your metabolic signals match your personal baseline beautifully. These actions are about maintaining this positive momentum.",
    actions: [
      { id: "stable-1", title: "Keep up the great work", description: "Your current routine is working for your metabolism. Keep doing what you did today." },
      { id: "stable-2", title: "Log your successful habits", description: "Make a mental note of what you ate and how you slept — these are the building blocks of your stable baseline." }
    ],
    monitoring: {
      title: "Routine check-in",
      description: "No specific action needed. Zyntra will continue monitoring your baseline in the background."
    }
  }
};

export function RecommendationsView({ glucoseScenario }: Props) {
  const content = SCENARIO_ACTIONS[glucoseScenario];

  return (
    <div className="pb-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-serif font-bold text-slate-900 mb-2">Today's Focus</h1>
        <p className="text-slate-600 text-[15px] leading-relaxed">
          Based on your current {glucoseScenario.toUpperCase()} scenario, here are the safest, most effective micro-actions you can take right now.
        </p>
      </div>

      <div className="bg-teal-50 rounded-[1.5rem] p-6 border border-teal-100 mb-10 shadow-sm">
        <h3 className="font-bold text-teal-900 mb-2 flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          Why these were suggested
        </h3>
        <p className="text-teal-800 text-sm leading-relaxed">
          {content.why}
        </p>
      </div>

      <h2 className="text-xl font-serif font-bold text-slate-900 mb-4">Small wins for the next 2 hours</h2>
      <div className="space-y-4 mb-10">
        {content.actions.map((rec) => (
          <ActionCard 
            key={rec.id}
            title={rec.title}
            description={rec.description}
            icon={<svg className="w-5 h-5 text-zyntra-teal" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" /></svg>}
          />
        ))}
      </div>

      <h2 className="text-xl font-serif font-bold text-slate-900 mb-4">Monitoring advice</h2>
      <div className="bg-white rounded-[1.5rem] p-6 shadow-sm border border-slate-100 flex items-start gap-4">
        <div className="w-10 h-10 bg-slate-50 rounded-full flex items-center justify-center flex-shrink-0 text-zyntra-navy border border-slate-100">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
        </div>
        <div>
          <h3 className="font-bold text-slate-900 mb-1">{content.monitoring.title}</h3>
          <p className="text-slate-500 text-sm leading-relaxed">
            {content.monitoring.description}
          </p>
        </div>
      </div>
    </div>
  );
}
