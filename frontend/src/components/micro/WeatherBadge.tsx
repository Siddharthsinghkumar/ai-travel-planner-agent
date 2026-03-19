type WeatherStatus = "rain" | "cloud" | "clear";

type Props = {
  status: WeatherStatus;
};

const styles: Record<WeatherStatus, { label: string; className: string }> = {
  rain: {
    label: "Rain",
    className: "bg-blue-500/20 text-blue-300",
  },
  cloud: {
    label: "Cloudy",
    className: "bg-gray-500/20 text-gray-300",
  },
  clear: {
    label: "Clear",
    className: "bg-yellow-400/20 text-yellow-300",
  },
};

const icons: Record<WeatherStatus, string> = {
  rain: "☔",
  cloud: "☁",
  clear: "☀",
};

export default function WeatherBadge({ status }: Props) {
  const cfg = styles[status];
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium ${cfg.className}`}>
      {icons[status]} {cfg.label}
    </span>
  );
}
