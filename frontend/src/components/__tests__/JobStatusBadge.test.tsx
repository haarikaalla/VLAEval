import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { JobStatusBadge } from "@/components/JobStatusBadge";

describe("JobStatusBadge", () => {
  it("renders the status text", () => {
    render(<JobStatusBadge status="succeeded" />);
    expect(screen.getByText("succeeded")).toBeInTheDocument();
  });
});
