import axios from "axios";

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});

// Attach the API key (dev/local use only -- browsers should prefer the
// short-lived JWT flow via /auth/token for anything user-facing).
apiClient.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem("vla_eval_api_key");
  if (apiKey) {
    config.headers = config.headers ?? {};
    config.headers["X-API-Key"] = apiKey;
  }
  return config;
});
