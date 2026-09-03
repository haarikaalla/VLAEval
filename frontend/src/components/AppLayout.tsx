import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";

const navItems = [
  { to: "/", label: "Leaderboard", end: true },
  { to: "/datasets", label: "Datasets" },
  { to: "/training", label: "Training Jobs" },
  { to: "/evaluation", label: "Evaluation Jobs" },
];

export function AppLayout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-brand-700 text-white shadow">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">VLA-Eval</h1>
          <nav className="flex gap-4">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  clsx(
                    "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                    isActive ? "bg-white text-brand-700" : "text-brand-50 hover:bg-brand-600",
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-8">
        <Outlet />
      </main>
      <footer className="text-center text-xs text-slate-400 py-4">
        VLA-Eval &mdash; Vision-Language-Action benchmarking platform
      </footer>
    </div>
  );
}
