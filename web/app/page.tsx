import { Play, Sparkles } from "lucide-react";

function GithubIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
    </svg>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0a0a0f] text-white overflow-x-hidden">

      {/* ── Nav ─────────────────────────────────────── */}
      <nav className="fixed top-0 w-full z-50 bg-[#0a0a0f]/80 backdrop-blur-md border-b border-white/5">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎓</span>
            <span className="font-bold text-lg tracking-tight">Gurukul AI</span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/LakshmiSravyaVedantham/gurukul-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition text-sm font-medium"
            >
              <GithubIcon size={16} />
              GitHub
            </a>
            <a
              href="https://www.youtube.com/@GurukuIAI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 transition text-sm font-medium"
            >
              <Play size={16} fill="white" />
              YouTube
            </a>
          </div>
        </div>
      </nav>

      {/* ── Hero ────────────────────────────────────── */}
      <section className="pt-40 pb-24 px-6 text-center relative">
        <div className="absolute top-20 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-purple-600/20 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute top-32 left-1/2 -translate-x-1/2 w-[300px] h-[150px] bg-yellow-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-300 text-sm font-medium mb-6">
            <Sparkles size={14} />
            100% Free · 100% Local · Apple Silicon
          </div>

          <h1 className="text-6xl sm:text-7xl font-black tracking-tight mb-6 leading-none">
            Kids&apos; Educational<br />
            <span className="bg-gradient-to-r from-yellow-400 via-orange-400 to-pink-500 bg-clip-text text-transparent">
              Videos, on Autopilot
            </span>
          </h1>

          <p className="text-xl text-white/60 max-w-2xl mx-auto mb-10 leading-relaxed">
            Gurukul AI turns any topic into a Pixar-style animated educational video —
            script, images, narration, animation, subtitles — fully local, zero cloud costs.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <a
              href="https://github.com/LakshmiSravyaVedantham/gurukul-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-black font-semibold hover:bg-white/90 transition text-base"
            >
              <GithubIcon size={18} />
              Get the Code
            </a>
            <a
              href="https://www.youtube.com/@GurukuIAI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold transition text-base"
            >
              <Play size={18} fill="white" />
              Watch on YouTube
            </a>
          </div>
        </div>
      </section>

      {/* ── Pipeline steps ──────────────────────────── */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold mb-3">From Topic to Video in 6 Steps</h2>
            <p className="text-white/50">Everything runs on your Mac. No API keys. No subscriptions.</p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { icon: "🌐", step: "01", label: "Research", desc: "DuckDuckGo + Wikipedia facts", color: "from-blue-500/20 to-blue-600/10" },
              { icon: "🧠", step: "02", label: "Script", desc: "Gemma 3 4B writes 10 scenes", color: "from-purple-500/20 to-purple-600/10" },
              { icon: "🖼️", step: "03", label: "Images", desc: "FLUX Dev generates landscapes", color: "from-orange-500/20 to-orange-600/10" },
              { icon: "🔊", step: "04", label: "Audio", desc: "Kokoro TTS narration", color: "from-green-500/20 to-green-600/10" },
              { icon: "🎬", step: "05", label: "Animate", desc: "LTX / Wan2.2 video models", color: "from-pink-500/20 to-pink-600/10" },
              { icon: "✨", step: "06", label: "Polish", desc: "Subtitles + xfade transitions", color: "from-yellow-500/20 to-yellow-600/10" },
            ].map((s) => (
              <div key={s.step} className={`rounded-2xl bg-gradient-to-b ${s.color} border border-white/5 p-5 flex flex-col items-center text-center gap-2`}>
                <span className="text-3xl">{s.icon}</span>
                <span className="text-xs text-white/30 font-mono">{s.step}</span>
                <span className="font-bold text-sm">{s.label}</span>
                <span className="text-xs text-white/50 leading-relaxed">{s.desc}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── selfimprove highlight ────────────────────── */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="rounded-3xl bg-gradient-to-br from-purple-900/40 to-pink-900/20 border border-purple-500/20 p-10">
            <div className="flex items-center gap-3 mb-5">
              <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center text-xl">⚡</div>
              <div>
                <div className="font-bold text-lg">/selfimprove — Agentic Pipeline</div>
                <div className="text-sm text-white/40">Auto-escalates models until quality passes</div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 mb-8 text-sm">
              {[
                { icon: "🎭", label: "Director", sub: "Gemma 4 expands prompt" },
                { icon: "🎬", label: "Creator", sub: "Generates video" },
                { icon: "🔍", label: "Critic", sub: "Qwen2.5-VL scores 1–10" },
                { icon: "🔄", label: "Refiner", sub: "Auto-retry better model" },
                { icon: "✨", label: "Polisher", sub: "Topaz 4K upscale" },
              ].map((item, i) => (
                <div key={i} className="flex items-center gap-2">
                  {i > 0 && <span className="text-white/20 text-lg">→</span>}
                  <div className="flex flex-col items-center gap-1 px-3 py-2 rounded-xl bg-white/5">
                    <span>{item.icon} {item.label}</span>
                    <span className="text-white/40 text-xs">{item.sub}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="grid sm:grid-cols-3 gap-4 text-sm">
              {[
                { label: "Model escalation", value: "LTX-2B → Wan2.2 Fun-5B → LTX-13B → Wan2.2 I2V-A14B" },
                { label: "Quality threshold", value: "Configurable min score (default 7.0/10)" },
                { label: "Output", value: "Training dataset + model leaderboard" },
              ].map((item) => (
                <div key={item.label} className="rounded-xl bg-white/5 p-4">
                  <div className="text-white/40 text-xs mb-1">{item.label}</div>
                  <div className="font-medium text-sm">{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Model grid ──────────────────────────────── */}
      <section className="py-20 px-6 bg-white/[0.02]">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold mb-3">9 Animation Models Built In</h2>
            <p className="text-white/50">From instant Ken Burns to 20-min cinematic Wan2.2 — pick your quality/speed tradeoff</p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { name: "Ken Burns", time: "< 5s", stars: 2, badge: "INSTANT", desc: "ffmpeg zoom+pan, zero GPU", color: "text-gray-400" },
              { name: "LTX Video 2B", time: "~40s", stars: 3, badge: "FAST", desc: "ComfyUI, great for iteration", color: "text-blue-400" },
              { name: "LTX Video 13B", time: "~11min", stars: 4, badge: "", desc: "High quality landscapes", color: "text-blue-400" },
              { name: "LTX-2.3 22B GGUF", time: "~4-6min", stars: 4, badge: "NEW", desc: "Newest LTX, speed+quality", color: "text-purple-400" },
              { name: "Wan2.2 Fun-5B GGUF", time: "~8-10min", stars: 4, badge: "GGUF", desc: "Object motion, stable", color: "text-orange-400" },
              { name: "Wan2.2 I2V-A14B GGUF", time: "~15-20min", stars: 5, badge: "BEST", desc: "Hero shots, dual-stage", color: "text-yellow-400" },
            ].map((m) => (
              <div key={m.name} className="rounded-2xl border border-white/5 bg-white/[0.03] p-5 hover:bg-white/[0.06] transition">
                <div className="flex items-start justify-between mb-3">
                  <span className={`font-bold text-sm ${m.color}`}>{m.name}</span>
                  {m.badge && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/60 font-mono">
                      {m.badge}
                    </span>
                  )}
                </div>
                <div className="text-white/40 text-xs mb-3">{m.desc}</div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-white/30 font-mono">{m.time}/scene</span>
                  <span className="text-yellow-400 text-xs">{"★".repeat(m.stars)}{"☆".repeat(5 - m.stars)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Free features strip ─────────────────────── */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-3">Stolen from OpenMontage. Made Free.</h2>
            <p className="text-white/50">Three premium features, zero cost, fully local</p>
          </div>
          <div className="grid sm:grid-cols-3 gap-6">
            {[
              {
                icon: "🔤",
                title: "Word-Level Subtitles",
                desc: "mlx-whisper transcribes on-device. Karaoke-style yellow highlight on current word. No OpenAI key.",
                tag: "WhisperX → mlx-whisper",
              },
              {
                icon: "🎞️",
                title: "Animated Transitions",
                desc: "ffmpeg xfade between every scene — dissolve, wipeleft, slideright. No Remotion, no JS bundle.",
                tag: "Remotion → ffmpeg",
              },
              {
                icon: "🌐",
                title: "Research-First Scripts",
                desc: "DuckDuckGo + Wikipedia facts fed to Gemma before writing. Episodes are accurate, not hallucinated.",
                tag: "Web research → free",
              },
            ].map((f) => (
              <div key={f.title} className="rounded-2xl border border-white/5 bg-white/[0.03] p-6">
                <span className="text-4xl mb-4 block">{f.icon}</span>
                <h3 className="font-bold mb-2">{f.title}</h3>
                <p className="text-white/50 text-sm leading-relaxed mb-4">{f.desc}</p>
                <span className="text-xs font-mono text-white/25 bg-white/5 px-2 py-1 rounded">{f.tag}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Hardware callout ─────────────────────────── */}
      <section className="py-12 px-6">
        <div className="max-w-3xl mx-auto text-center rounded-2xl border border-white/5 bg-white/[0.02] p-8">
          <div className="text-4xl mb-4">🖥️</div>
          <h3 className="font-bold text-lg mb-2">Optimised for Mac Studio M4 Max 36GB</h3>
          <p className="text-white/50 text-sm">
            All models tested on Apple Silicon MPS. Sequential generation to avoid memory spikes.
            FLUX Schnell for previews, FLUX Dev for finals. GGUF quantisation for stability over BF16.
          </p>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────── */}
      <section className="py-24 px-6 text-center">
        <div className="max-w-2xl mx-auto">
          <h2 className="text-4xl font-black mb-4">Start Making Videos</h2>
          <p className="text-white/50 mb-8">Clone the repo, start the Gradio app, type a topic.</p>
          <div className="bg-[#111118] rounded-xl p-5 text-left font-mono text-sm text-green-400 mb-8 border border-white/5 leading-relaxed">
            <div><span className="text-white/30">$ </span>git clone https://github.com/LakshmiSravyaVedantham/gurukul-ai</div>
            <div><span className="text-white/30">$ </span>cd gurukul-ai &amp;&amp; pip install -r requirements.txt</div>
            <div><span className="text-white/30">$ </span>python app.py<span className="text-white/30">  # open http://localhost:7860</span></div>
          </div>
          <div className="flex flex-wrap justify-center gap-4">
            <a
              href="https://github.com/LakshmiSravyaVedantham/gurukul-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-black font-semibold hover:bg-white/90 transition"
            >
              <GithubIcon size={18} />
              View on GitHub
            </a>
            <a
              href="https://www.youtube.com/@GurukuIAI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 rounded-xl border border-white/10 hover:bg-white/5 transition"
            >
              <Play size={18} />
              Watch Videos
            </a>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────── */}
      <footer className="border-t border-white/5 py-8 px-6 text-center text-white/25 text-sm">
        <p>Gurukul AI · Kids Educational Video Pipeline · Fully Local · Open Source</p>
      </footer>

    </main>
  );
}
