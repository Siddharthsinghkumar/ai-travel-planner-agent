type Props = {
  score: number;
};

function getScoreColor(score: number) {
  if (score > 90) return "#34d399";
  if (score > 75) return "#facc15";
  return "#f87171";
}

export default function ScoreRing({ score }: Props) {
  const clamped = Math.max(0, Math.min(100, score));
  const radius = 20;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;
  const color = getScoreColor(clamped);

  return (
    <div className="flex items-center gap-2">
      <svg width="48" height="48" viewBox="0 0 48 48" className="drop-shadow-[0_0_6px_currentColor]" style={{ color }}>
        <circle cx="24" cy="24" r={radius} fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="4" />
        <circle
          cx="24"
          cy="24"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 24 24)"
        />
        <text x="24" y="28" textAnchor="middle" fontSize="11" fontWeight="700" fill="currentColor">
          {Math.round(clamped)}
        </text>
      </svg>
    </div>
  );
}
