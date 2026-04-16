"use client";
import { useState } from "react";

// ── Feature card ────────────────────────────────────────────────────────────

const cardThemes = [
  { 
    bg: "#F5E9C8", 
    robot: <img src="/images/thinking.png" alt="Thinking Mascot" style={{ width: 120, height: 130, objectFit: 'contain' }} /> 
  },
  { 
    bg: "#EDE4D8", 
    robot: <img src="/images/studying.png" alt="Studying Mascot" style={{ width: 120, height: 130, objectFit: 'contain' }} /> 
  },
  { 
    bg: "#EAE4DC", 
    robot: <img src="/images/watching_tv.png" alt="Watching TV Mascot" style={{ width: 130, height: 130, objectFit: 'contain' }} /> 
  },
];

function FeatureCard({ title, description, index, onGetStarted }: { title: string, description: string, index: number, onGetStarted?: (t: string) => void }) {
  const { bg, robot } = cardThemes[index % cardThemes.length];
  return (
    <div style={{
      background: bg,
      borderRadius: 20,
      padding: "24px 24px 20px",
      display: "flex",
      flexDirection: "column",
      minHeight: 280,
      position: "relative",
      overflow: "hidden",
      flex: 1,
    }}>
      <h2 style={{ fontSize: 22, fontWeight: 600, color: "#2D2438", margin: "0 0 10px" }}>{title}</h2>
      <p style={{ fontSize: 14, color: "#5A4E6B", lineHeight: 1.55, margin: "0 0 auto", maxWidth: 200 }}>{description}</p>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginTop: 16 }}>
        <div style={{ marginBottom: -4 }}>{robot}</div>
        <button
          onClick={() => onGetStarted?.(title)}
          style={{
            background: "#7B5CF0",
            color: "white",
            border: "none",
            borderRadius: 50,
            padding: "12px 22px",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
            whiteSpace: "nowrap",
            alignSelf: "flex-end",
            transition: "background 0.15s",
          }}
          onMouseEnter={e => e.currentTarget.style.background = "#6748DC"}
          onMouseLeave={e => e.currentTarget.style.background = "#7B5CF0"}
        >
          Get Started
        </button>
      </div>
    </div>
  );
}

// ── Stat card ───────────────────────────────────────────────────────────────

function StatCard({ value, label }: { value: number | string, label: string }) {
  return (
    <div style={{
      background: "#EBF2FA",
      borderRadius: 16,
      padding: "28px 20px",
      flex: 1,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 10,
    }}>
      <span style={{ fontSize: 42, fontWeight: 800, color: "#2D2438" }}>{value}</span>
      <span style={{ fontSize: 13, color: "#7A6E8A", textAlign: "center", lineHeight: 1.4 }}>{label}</span>
    </div>
  );
}

// ── Welcome / Aster widget ──────────────────────────────────────────────────

function WelcomeWidget() {
  const [inputVal, setInputVal] = useState("");
  return (
    <div style={{
      background: "#C4B0F0",
      borderRadius: 20,
      padding: "22px 24px",
      display: "flex",
      flexDirection: "column",
      gap: 10,
      minWidth: 280,
    }}>
      <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#1E1040" }}>
        Welcome to Sparkle Labs
      </h3>
      <p style={{ margin: 0, fontSize: 13, color: "#3D2880" }}>I am Aster your assistant!</p>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 12, marginTop: 4 }}>
        <input
          type="text"
          placeholder="Take me to the parents questionnaire..."
          value={inputVal}
          onChange={e => setInputVal(e.target.value)}
          style={{
            flex: 1,
            background: "rgba(255,255,255,0.65)",
            border: "none",
            borderRadius: 12,
            padding: "12px 14px",
            fontSize: 13,
            color: "#2D2438",
            outline: "none",
          }}
        />
        <div style={{ flexShrink: 0, marginBottom: -8 }}>
          <img 
            src="/images/mascot.png" 
            alt="Aster Mascot" 
            style={{ width: 100, height: 120, objectFit: 'contain' }} 
          />
        </div>
      </div>
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

const FEATURES = [
  {
    title: "ParentLens",
    description: "Parent questionnaire that turns home observations into clear ASD support insights.",
  },
  {
    title: "NeuroScan",
    description: "AI-powered analysis of EEG and videos to assess autism-related brain patterns.",
  },
  {
    title: "QuestScope",
    description: "Adventure-based learning for kids with built-in progress tracking for caregivers.",
  },
];

const STATS = [
  { value: 6,  label: "Monitoring\nSessions" },
  { value: 7,  label: "Completed\nQuestionnaires" },
  { value: 0,  label: "EEG Analysis" },
];

export default function SparkleLabsHome() {
  return (
    <div style={{
      minHeight: "100vh",
      background: "#FAF7F2",
      fontFamily: "'Nunito', 'Poppins', system-ui, sans-serif",
      padding: "0 0 40px",
    }}>

      {/* ── Header ── */}
      <header style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "18px 32px",
        background: "white",
        borderBottom: "1px solid #F0EBE0",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <img src="/images/puzzle.png" alt="Puzzle Logo" style={{ width: 32, height: 32, objectFit: 'contain' }} />
          <span style={{ fontSize: 22, fontWeight: 800, color: "#7B5CF0" }}>Sparkle Labs</span>
        </div>
        <div style={{
          width: 40, height: 40,
          borderRadius: "50%",
          background: "#D8D0E8",
        }} />
      </header>

      {/* ── Main content ── */}
      <main style={{ padding: "28px 32px", display: "flex", flexDirection: "column", gap: 24 }}>

        {/* Feature cards row */}
        <div style={{ display: "flex", gap: 20 }}>
          {FEATURES.map((f, i) => (
            <FeatureCard
              key={f.title}
              title={f.title}
              description={f.description}
              index={i}
              onGetStarted={title => alert(`Navigating to ${title}…`)}
            />
          ))}
        </div>

        {/* Quickstats + Welcome row */}
        <div style={{ display: "flex", gap: 20, alignItems: "stretch" }}>

          {/* Quickstats */}
          <div style={{
            background: "#EEE5E5",
            borderRadius: 20,
            padding: "24px 28px",
            flex: 1,
          }}>
            <h2 style={{ margin: "0 0 18px", fontSize: 20, fontWeight: 600, color: "#2D2438" }}>
              Quickstats
            </h2>
            <div style={{ display: "flex", gap: 16 }}>
              {STATS.map(s => (
                <StatCard key={s.label} value={s.value} label={s.label} />
              ))}
            </div>
          </div>

          {/* Welcome widget */}
          <WelcomeWidget />
        </div>
      </main>
    </div>
  );
}
