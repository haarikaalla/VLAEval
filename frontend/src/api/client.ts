import axios from "axios";

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});

// Browser requests use a short-lived token. The long-lived API key is only
// sent once by exchangeApiKey and is never persisted in localStorage.
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem("vla_eval_access_token");
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
