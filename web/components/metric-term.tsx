type Props = {
  label: string;
  help: string;
};

export function MetricTerm({ label, help }: Props) {
  return (
    <span
      className="metric-term"
      tabIndex={0}
      aria-label={`${label}: ${help}`}
      data-help={help}
    >
      {label}
      <span aria-hidden="true">?</span>
    </span>
  );
}
