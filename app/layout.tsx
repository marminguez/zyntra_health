import type { Metadata } from "next";
import { Inter, Fraunces } from 'next/font/google';
import Providers from "./providers";
import "./globals.css";

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const fraunces = Fraunces({ subsets: ['latin'], variable: '--font-fraunces' });

export const metadata: Metadata = {
    title: "Zyntra – Adaptive Metabolic Intelligence",
    description:
        "Secure infrastructure layer for metabolic and wearable signal ingestion, baseline computation, and predictive risk scoring.",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
            <body className="font-sans antialiased text-slate-900 bg-slate-50 min-h-screen">
                <Providers>{children}</Providers>
            </body>
        </html>
    );
}
