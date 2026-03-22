// ============================================================
// Breadcrumb Navigation – Shows drilldown path
// ============================================================

import { ChevronRight } from 'lucide-react';
import { useStore } from '../../store/useStore';

export function Breadcrumb() {
  const drilldown = useStore(s => s.drilldown);
  const setDrilldown = useStore(s => s.setDrilldown);

  const crumbs: { label: string; onClick: () => void }[] = [
    {
      label: 'Pipeline DAG',
      onClick: () => setDrilldown({ level: 'pipeline', selectedPhase: undefined, selectedTreeNode: undefined, selectedSubagent: undefined, iteration: drilldown.iteration }),
    },
  ];

  if (drilldown.level === 'tree' || drilldown.level === 'subagent' || drilldown.level === 'condition') {
    crumbs.push({
      label: 'Research Tree',
      onClick: () => setDrilldown({ level: 'tree', selectedPhase: 'tree_research', selectedTreeNode: undefined, selectedSubagent: undefined, iteration: drilldown.iteration }),
    });
  }

  if (drilldown.level === 'subagent' || drilldown.level === 'condition') {
    crumbs.push({
      label: 'Subagent Trace',
      onClick: () => setDrilldown({ ...drilldown, level: 'subagent', selectedSubagent: undefined }),
    });
  }

  if (drilldown.level === 'condition') {
    crumbs.push({
      label: 'Conditions',
      onClick: () => {},
    });
  }

  return (
    <nav className="flex items-center gap-1 text-xs">
      {crumbs.map((crumb, i) => {
        const isLast = i === crumbs.length - 1;
        return (
          <span key={i} className="flex items-center gap-1">
            {i > 0 && <ChevronRight className="w-3 h-3 text-gray-600" />}
            <button
              onClick={crumb.onClick}
              className={`px-1.5 py-0.5 rounded transition-colors ${
                isLast
                  ? 'text-white bg-gray-700/50'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {crumb.label}
            </button>
          </span>
        );
      })}
    </nav>
  );
}
