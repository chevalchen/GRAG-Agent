import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SuggestChips } from "../components/SuggestChips";

const mockSend = vi.fn();

vi.mock("../store/chatStore", () => ({
  useChatStore: (sel: (s: { isStreaming: boolean; sendMessage: () => void }) => unknown) =>
    sel({ isStreaming: false, sendMessage: mockSend }),
}));

describe("SuggestChips", () => {
  it("renders nothing when questions array is empty", () => {
    const { container } = render(<SuggestChips questions={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders a chip for each question", () => {
    render(<SuggestChips questions={["问题一", "问题二", "问题三"]} />);
    expect(screen.getByText("问题一")).toBeTruthy();
    expect(screen.getByText("问题二")).toBeTruthy();
    expect(screen.getByText("问题三")).toBeTruthy();
  });

  it("calls sendMessage with chip text on click", () => {
    render(<SuggestChips questions={["点击问题"]} />);
    fireEvent.click(screen.getByText("点击问题"));
    expect(mockSend).toHaveBeenCalledWith("点击问题");
  });
});
