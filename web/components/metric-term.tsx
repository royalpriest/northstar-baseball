type Props = {
  label: string;
  help: string;
};

export function MetricTerm({ label, help }: Props) {
  return (
    <abbr className="metric-term" title={help}>
      {label}
      <span aria-hidden="true">?</span>
    </abbr>
  );
}
