type Props = {
  data: number[];
};

export default function PriceSparkline({ data }: Props) {
  if (!data || data.length < 2) return null;

  const width = 100;
  const height = 30;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data
    .map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-label="Price trend">
      <polyline
        fill="none"
        stroke="rgba(56, 189, 248, 0.85)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
        opacity="0.8"
      />
    </svg>
  );
}
