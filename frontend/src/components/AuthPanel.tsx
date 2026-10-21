import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { exchangeApiKey } from "@/api/endpoints";

export function AuthPanel() {
  const [apiKey, setApiKey] = useState("");
  const [authenticated, setAuthenticated] = useState(
    () => Boolean(localStorage.getItem("vla_eval_access_token")),
  );
  const login = useMutation({
    mutationFn: exchangeApiKey,
    onSuccess: (token) => {
      localStorage.setItem("vla_eval_access_token", token.access_token);
      setApiKey("");
      setAuthenticated(true);
      window.location.reload();
    },
  });

  if (authenticated) {
    return (
      <button
        type="button"
        className="rounded-md border border-brand-300 px-3 py-1.5 text-sm hover:bg-brand-600"
        onClick={() => {
          localStorage.removeItem("vla_eval_access_token");
          setAuthenticated(false);
          window.location.reload();
        }}
      >
        Sign out
      </button>
    );
  }

  return (
    <form
      className="flex items-center gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        if (apiKey.trim()) login.mutate(apiKey.trim());
      }}
    >
      <label className="sr-only" htmlFor="api-key">API key</label>
      <input
        id="api-key"
        type="password"
        autoComplete="current-password"
        placeholder="API key"
        value={apiKey}
        onChange={(event) => setApiKey(event.target.value)}
        className="w-32 rounded-md border border-brand-400 bg-brand-800 px-2 py-1.5 text-sm text-white placeholder:text-brand-200"
      />
      <button
        type="submit"
        disabled={login.isPending}
        className="rounded-md bg-white px-3 py-1.5 text-sm font-medium text-brand-700 disabled:opacity-50"
      >
        {login.isPending ? "Signing in..." : "Sign in"}
      </button>
      {login.isError && <span className="text-xs text-red-200">Invalid key</span>}
    </form>
  );
}
