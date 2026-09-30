import { percentile } from "@/lib/format";
import { MetricTerm } from "@/components/metric-term";

type Props = {
  label: string;
  value?: number | null;
  detail?: string;
  help?: string;
  tone?: "blue" | "cyan" | "violet";
};

export function PercentileBar({
  label,
  value,
  detail,
  help,
  tone = "blue",
}: Props) {
  const clamped = Math.max(0, Math.min(100, value ?? 0));

  return (
    <div className="percentile-block">
      <div className="percentile-block__header">
        <div>
          <span className="eyebrow">
            {help ? <MetricTerm label={label} help={help} /> : label}
          </span>
          {detail ? <p>{detail}</p> : null}
        </div>
        <strong>{percentile(value)}</strong>
      </div>
      <div className="percentile-track" aria-hidden="true">
        <span
          className={`percentile-fill percentile-fill--${tone}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
