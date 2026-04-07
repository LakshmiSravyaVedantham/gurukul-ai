import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Gurukul AI — Kids Educational Video Pipeline",
  description: "Turn any topic into a Pixar-style animated educational video. Fully local, free, Apple Silicon. Script → Images → Audio → Animation → Subtitles.",
  openGraph: {
    title: "Gurukul AI — Kids Educational Video Pipeline",
    description: "100% local AI pipeline for kids' educational videos. No cloud. No API keys.",
    siteName: "Gurukul AI",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
