// Cold-start landing shown when there are no messages yet
import { PRESET_TOPICS } from "../lib/topics";
import { useChatStore } from "../store/chatStore";

// Category labels for each preset topic (purely decorative)
const TOPIC_CATEGORIES = ["方剂", "方剂", "辨证", "本草", "辨证", "养生", "调理", "方剂"];

// Decorative herb emblem SVG
function HerbEmblem() {
  return (
    <svg
      width="40"
      height="40"
      viewBox="0 0 40 40"
      fill="currentColor"
      aria-hidden="true"
    >
      {/* stem */}
      <path
        d="M20 37 V13"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
      />
      {/* left leaf — curves outward left from mid-stem and back */}
      <path d="M20 27 C15 25 8 20 8 13 C13 13 20 20 20 27 Z" opacity="0.9" />
      {/* right leaf — curves outward right toward tip and back */}
      <path d="M20 18 C25 15 31 11 30 5 C25 5 19 11 20 18 Z" opacity="0.9" />
      {/* bud at apex */}
      <circle cx="20" cy="11" r="2.5" opacity="0.7" />
    </svg>
  );
}

export function HomeHero() {
  const sendMessage = useChatStore((s) => s.sendMessage);

  return (
    <div className="relative flex flex-col items-center justify-center h-full px-6 text-center select-none overflow-hidden">

      {/* decorative watermark — giant 仁 character (benevolence in TCM philosophy) */}
      <div
        className="absolute inset-0 flex items-center justify-center pointer-events-none"
        aria-hidden="true"
      >
        <span
          className="font-serif leading-none text-jade select-none"
          style={{ fontSize: "280px", opacity: 0.025 }}
        >
          仁
        </span>
      </div>

      {/* decorative corner flourishes */}
      <div
        className="absolute top-6 left-6 text-accent/20 font-serif text-lg pointer-events-none"
        aria-hidden="true"
      >
        ✦
      </div>
      <div
        className="absolute top-6 right-6 text-accent/20 font-serif text-lg pointer-events-none"
        aria-hidden="true"
      >
        ✦
      </div>

      {/* central emblem */}
      <div className="relative mb-5 z-10">
        {/* outer ring */}
        <div className="w-20 h-20 rounded-full border border-accent/30 flex items-center justify-center mx-auto relative">
          {/* inner ring */}
          <div className="w-14 h-14 rounded-full border border-accent/20 bg-accent/8 flex items-center justify-center">
            <div className="text-accent">
              <HerbEmblem />
            </div>
          </div>
          {/* decorative dots on ring */}
          {[0, 90, 180, 270].map((deg) => (
            <span
              key={deg}
              className="absolute w-1 h-1 rounded-full bg-accent/40"
              style={{
                top: "50%",
                left: "50%",
                transform: `rotate(${deg}deg) translateY(-38px) translateX(-2px)`,
              }}
            />
          ))}
        </div>
      </div>

      {/* title */}
      <div className="relative z-10 mb-2">
        <h1 className="font-serif text-2xl font-semibold text-paper tracking-wide">
          中医药智能问诊助手
        </h1>
        <div className="gold-divider mt-2 mx-auto w-32" />
        <p className="text-muted text-sm mt-3 max-w-sm leading-relaxed">
          基于知识图谱的 AI 中医问答系统，提供专业的中药知识查询与健康建议。
        </p>
      </div>

      {/* disclaimer */}
      <p className="relative z-10 text-xs text-muted/60 mb-6 max-w-sm mt-3 flex items-center gap-1 justify-center">
        <span>⚠</span>
        <span>本系统仅供参考，不替代专业医疗诊断。请遵医嘱，谨慎用药。</span>
      </p>

      {/* preset topic cards */}
      <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
        {PRESET_TOPICS.map((topic, idx) => (
          <button
            key={topic}
            onClick={() => sendMessage(topic)}
            className="
              group text-left px-4 py-3 rounded-xl
              bg-surface-2 border border-surface-3
              hover:border-accent/40 hover:bg-surface-3
              transition-all duration-200
            "
          >
            {/* category badge */}
            <span className="
              inline-block text-[10px] text-jade/80 border border-jade/30
              rounded px-1.5 py-px mb-1.5 font-sans tracking-widest
              group-hover:border-jade/50 group-hover:text-jade
              transition-colors
            ">
              {TOPIC_CATEGORIES[idx] ?? "问诊"}
            </span>
            <p className="text-sm text-paper/80 group-hover:text-paper leading-snug transition-colors">
              {topic}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}
