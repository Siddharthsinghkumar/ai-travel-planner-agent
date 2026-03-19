import { useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import "./flights-ticker.css";

type Props = { items: string[]; speed?: number };

export default function FlightsTicker({ items, speed = 40 }: Props) {
  const renderedItems = useMemo(() => (items.length ? items : ["Travelyst"]), [items]);
  const trackRef = useRef<HTMLDivElement>(null);
  const [durationSec, setDurationSec] = useState(30);

  useLayoutEffect(() => {
    const track = trackRef.current;
    if (!track) return;

    const updateDuration = () => {
      const fullWidth = track.scrollWidth;
      const loopDistance = fullWidth / 2;
      if (loopDistance <= 0 || speed <= 0) {
        setDurationSec(30);
        return;
      }
      setDurationSec(Math.max(loopDistance / speed, 8));
    };

    updateDuration();
    const ro = new ResizeObserver(updateDuration);
    ro.observe(track);
    return () => ro.disconnect();
  }, [renderedItems, speed]);

  const tickerStyle = {
    "--speed": `${speed}px`,
    "--ticker-length": `${durationSec}s`,
  } as CSSProperties;

  return (
    <div className="ticker-wrap" aria-hidden="true" onMouseDown={(e) => e.preventDefault()} tabIndex={-1}>
      <div
        ref={trackRef}
        className="ticker-track"
        style={tickerStyle}
        role="presentation"
      >
        <div className="ticker-group" aria-hidden="true">
          {renderedItems.map((t, i) => (
            <div className="ticker-item" key={`a-${i}-${t}`} tabIndex={-1}>
              {t}
            </div>
          ))}
        </div>
        <div className="ticker-group" aria-hidden="true">
          {renderedItems.map((t, i) => (
            <div className="ticker-item" key={`b-${i}-${t}`} tabIndex={-1}>
              {t}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
