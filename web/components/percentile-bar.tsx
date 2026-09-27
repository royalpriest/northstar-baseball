import { percentile } from "@/lib/format";

type Props = {
  label: string;
  value?: number | null;
  detail?: string;
  tone?: "blue" | "cyan" | "violet";
};

export function PercentileBar({
  label,
  value,
  detail,
  tone = "blue",
}: Props) {
  const clamped = Math.max(0, Math.min(100, value ?? 0));

  return (
    <div className="percentile-block">
      <div className="percentile-block__header">
        <div>
          <span className="eyebrow">{label}</span>
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
