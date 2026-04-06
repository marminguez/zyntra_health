"use client";

import { signIn } from "next-auth/react";
import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
    const router = useRouter();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    async function handleSubmit(e: FormEvent) {
        e.preventDefault();
        setError("");
        setLoading(true);

        const res = await signIn("credentials", {
            redirect: false,
            email,
            password,
        });

        setLoading(false);

        if (res?.error) {
            setError("Invalid credentials. Use password: hackathon-dev-only");
        } else {
            router.push("/dashboard");
        }
    }

    return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-zyntra-navy px-4 font-sans relative">
            
            {/* Logo Section */}
            <div className="text-center mb-8">
                <h1 className="text-5xl font-serif font-bold text-white mb-3">Zyntra</h1>
                <p className="text-slate-400 text-sm tracking-wide">Your metabolic health, personalised.</p>
            </div>

            {/* Login Card */}
            <div className="bg-white rounded-[2rem] p-8 sm:p-10 w-full max-w-md shadow-2xl relative z-10">
                <h2 className="text-3xl font-serif font-bold text-slate-900 mb-2">Welcome back</h2>
                <p className="text-slate-600 mb-8">Please enter your details to continue.</p>

                <form onSubmit={handleSubmit} className="space-y-6">
                    
                    {/* Email Input */}
                    <div className="relative">
                        <label 
                            htmlFor="email" 
                            className="absolute -top-2.5 left-4 bg-white px-1 text-xs font-bold text-slate-900"
                        >
                            Email Address
                        </label>
                        <input
                            id="email"
                            type="email"
                            placeholder="name@clinical.com"
                            className="w-full px-4 py-3.5 rounded-xl border border-slate-200 focus:outline-none focus:border-zyntra-navy transition-colors text-slate-900"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>

                    {/* Password Input */}
                    <div className="relative">
                        <label 
                            htmlFor="password" 
                            className="absolute -top-2.5 left-4 bg-white px-1 text-xs font-bold text-slate-900"
                        >
                            Password
                        </label>
                        <input
                            id="password"
                            type="password"
                            placeholder="••••••••"
                            className="w-full px-4 py-3.5 rounded-xl border border-slate-200 focus:outline-none focus:border-zyntra-navy transition-colors text-slate-900"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    {error && <p className="text-red-500 text-sm text-center">{error}</p>}

                    {/* Submit Button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-zyntra-teal hover:bg-zyntra-teal-hover text-zyntra-green font-bold text-lg py-3.5 rounded-xl transition-colors mt-2"
                    >
                        {loading ? "Signing in..." : "Sign in"}
                    </button>
                    
                    {/* Forgot Password */}
                    <div className="text-center mt-6">
                        <a href="#" className="text-slate-800 hover:text-zyntra-navy font-medium text-sm">
                            Forgot password?
                        </a>
                    </div>
                    
                    {/* Divider */}
                    <div className="relative flex items-center justify-center my-6">
                        <div className="absolute inset-0 flex items-center">
                            <div className="w-full border-t border-slate-200"></div>
                        </div>
                        <div className="relative bg-white px-4">
                            <span className="text-[10px] font-bold tracking-widest text-slate-900 uppercase">
                                New to Zyntra?
                            </span>
                        </div>
                    </div>

                    {/* Request Access */}
                    <div className="text-center text-sm text-slate-600">
                        Don't have an account?{' '}
                        <a href="#" className="text-zyntra-green font-bold hover:underline">
                            Request Access
                        </a>
                    </div>
                </form>
            </div>

            {/* Footer */}
            <div className="absolute bottom-8 flex items-center gap-2 text-slate-500 text-xs font-bold tracking-widest uppercase">
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 1.944A11.954 11.954 0 012.166 5C2.056 5.649 2 6.319 2 7c0 5.225 3.34 9.67 8 11.317C14.66 16.67 18 12.225 18 7c0-.682-.057-1.35-.166-2.001A11.954 11.954 0 0110 1.944zM11 14a1 1 0 11-2 0 1 1 0 012 0zm0-7a1 1 0 10-2 0v3a1 1 0 102 0V7z" clipRule="evenodd" />
                </svg>
                Secure Laboratory Standards
            </div>
        </div>
    );
}
