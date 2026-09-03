import { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppLayout } from "@/components/AppLayout";
import { LeaderboardPage } from "@/pages/LeaderboardPage";
import { DatasetsPage } from "@/pages/DatasetsPage";
import { TrainingPage } from "@/pages/TrainingPage";
import { EvaluationPage } from "@/pages/EvaluationPage";
import "@/index.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 5000 } },
});

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element (#root) not found in index.html");
}

ReactDOM.createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<LeaderboardPage />} />
            <Route path="datasets" element={<DatasetsPage />} />
            <Route path="training" element={<TrainingPage />} />
            <Route path="evaluation" element={<EvaluationPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
