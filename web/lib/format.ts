export function percentile(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const rounded = Math.max(0, Math.min(100, Math.round(value)));
  const mod100 = rounded % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${rounded}th`;
  switch (rounded % 10) {
    case 1:
      return `${rounded}st`;
    case 2:
      return `${rounded}nd`;
    case 3:
      return `${rounded}rd`;
    default:
      return `${rounded}th`;
  }
}

export function decimal(value?: number | null, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits).replace(/^0\./, ".");
}

export function signed(value?: number | null, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const formatted = value.toFixed(digits);
  return value > 0 ? `+${formatted}` : formatted;
}

export function percentRate(value?: number | null, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function integer(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.round(value).toLocaleString("en-CA");
}

export function isMissing(value?: number | null): value is null | undefined {
  return value === null || value === undefined || Number.isNaN(value);
}

// Bare 0–100 percentile for compact axes; a dash means unavailable, not zero.
export function percentileValue(value?: number | null): string {
  if (isMissing(value)) return "—";
  return String(Math.max(0, Math.min(100, Math.round(value))));
}

export function barWidth(value?: number | null): string {
  if (isMissing(value)) return "0%";
  return `${Math.max(0, Math.min(100, value))}%`;
}
