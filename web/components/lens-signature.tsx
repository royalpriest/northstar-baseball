import { isMissing, percentile } from "@/lib/format";

type Lens = {
  key: "selection" | "contact" | "damage";
  label: string;
  value?: number | null;
  detail?: string;
};

type Props = {
  selection?: number | null;
  contact?: number | null;
  damage?: number | null;
  selectionDetail?: string;
  contactDetail?: string;
  damageDetail?: string;
  compact?: boolean;
  className?: string;
};

function clamp(value?: number | null) {
  return Math.max(0, Math.min(100, value ?? 0));
}

export function LensSignature({
  selection,
  contact,
  damage,
  selectionDetail,
  contactDetail,
  damageDetail,
  compact = false,
  className = "",
}: Props) {
  const lenses: Lens[] = [
    {
      key: "selection",
      label: "Selection",
      value: selection,
      detail: selectionDetail,
    },
    {
      key: "contact",
      label: "Contact",
      value: contact,
      detail: contactDetail,
    },
    {
      key: "damage",
      label: "Damage",
      value: damage,
      detail: damageDetail,
    },
  ];

  return (
    <div
      className={`lens-signature ${compact ? "lens-signature--compact" : ""} ${className}`.trim()}
      aria-label="NorthStar three-lens profile"
    >
      {lenses.map((lens) => {
        const available = !isMissing(lens.value);
        const width = clamp(lens.value);
        return (
          <div className={`lens-line lens-line--${lens.key}`} key={lens.key}>
            <div className="lens-line__meta">
              <span>{lens.label}</span>
              {lens.detail ? <small>{lens.detail}</small> : null}
              <strong>{percentile(lens.value)}</strong>
            </div>
            <div className="lens-line__track" aria-hidden="true">
              <span className="lens-line__midpoint" />
              {available ? (
                <>
                  <span className="lens-line__fill" style={{ width: `${width}%` }} />
                  <span className="lens-line__marker" style={{ left: `${width}%` }}>
                    <i />
                  </span>
                </>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}
