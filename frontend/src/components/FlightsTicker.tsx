import { useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import "./flights-ticker.css";

type Props = { items: string[]; speed?: number };

export default function FlightsTicker({ items, speed = 40 }: Props) {
  const renderedItems = useMemo(() => (items.length ? items : ["Travelyst"]), [items]);
  const groupRef = useRef<HTMLDivElement>(null);
  const [loopDistancePx, setLoopDistancePx] = useState(0);
  const [durationSec, setDurationSec] = useState(30);

  useLayoutEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    let rafA = 0;
    let rafB = 0;

    const updateDuration = () => {
      const loopDistance = group.getBoundingClientRect().width;
      if (loopDistance <= 0 || speed <= 0) {
        setLoopDistancePx(0);
        setDurationSec(30);
        return;
      }
      setLoopDistancePx(loopDistance);
      setDurationSec(Math.max(loopDistance / speed, 8));
    };

    rafA = window.requestAnimationFrame(() => {
      rafB = window.requestAnimationFrame(updateDuration);
    });
    const ro = new ResizeObserver(updateDuration);
    ro.observe(group);
    document.fonts?.ready.then(updateDuration).catch(() => undefined);
    return () => {
      ro.disconnect();
      if (rafA) window.cancelAnimationFrame(rafA);
      if (rafB) window.cancelAnimationFrame(rafB);
    };
  }, [renderedItems, speed]);

  const tickerStyle = {
    "--ticker-loop-distance": `${loopDistancePx}px`,
    "--ticker-length": `${durationSec}s`,
  } as CSSProperties;

  return (
    <div className="ticker-wrap" aria-hidden="true" onMouseDown={(e) => e.preventDefault()} tabIndex={-1}>
      <div
        className="ticker-track"
        style={tickerStyle}
        role="presentation"
      >
        <div ref={groupRef} className="ticker-group" aria-hidden="true">
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
        <div className="ticker-group" aria-hidden="true">
          {renderedItems.map((t, i) => (
            <div className="ticker-item" key={`c-${i}-${t}`} tabIndex={-1}>
              {t}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
