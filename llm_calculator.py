import math
import tkinter as tk
from tkinter import ttk


MODEL_SIZE_OPTIONS = (
    "1",
    "3",
    "7",
    "8",
    "13",
    "14",
    "32",
    "70",
    "123",
    "405",
)

BIT_WIDTH_OPTIONS = ("1", "2", "3", "4", "5", "6", "7", "8", "16", "32")
CONTEXT_OPTIONS = ("2048", "4096", "8192", "16384", "32768", "65536", "128000")

QUANTIZATION_HELP = {
    "Q": "Generic low-bit quantization. A small overhead is added for scales/metadata.",
    "FP": "Floating-point weights. Effective bits stay equal to the selected bit width.",
    "INT": "Integer weights. Effective bits stay equal to the selected bit width.",
    "MXFP": "Microscaled floating point. A small overhead is added for shared scales.",
}


def parse_positive_float(value):
    number = float(value)
    if number <= 0:
        raise ValueError
    return number


def parse_positive_int(value):
    number = int(value)
    if number <= 0:
        raise ValueError
    return number


def format_bytes(num_bytes):
    gib = num_bytes / (1024 ** 3)
    if gib >= 1:
        return f"{gib:,.2f} GiB"

    mib = num_bytes / (1024 ** 2)
    if mib >= 1:
        return f"{mib:,.2f} MiB"

    kib = num_bytes / 1024
    return f"{kib:,.2f} KiB"


def effective_bits_per_parameter(family, bits):
    if family == "Q":
        return bits + 0.50 if bits < 16 else float(bits)
    if family == "MXFP":
        return bits + 0.25 if bits < 16 else float(bits)
    return float(bits)


def estimate_transformer_shape(params_billions):
    # Heuristic only. This keeps the KV estimate in a realistic range for common LLM sizes.
    layers = max(12, min(200, round(12 + 8 * math.sqrt(params_billions))))
    hidden_size = int(round((1536 * (params_billions ** 0.35) + 1024) / 128) * 128)
    return layers, hidden_size


def estimate_memory(params_billions, family, bits, context_tokens):
    effective_bits = effective_bits_per_parameter(family, bits)
    parameter_count = params_billions * 1_000_000_000
    weights_bytes = parameter_count * effective_bits / 8

    layers, hidden_size = estimate_transformer_shape(params_billions)

    # Assumes fp16/bf16 KV cache: 2 bytes for keys and 2 bytes for values.
    kv_bytes_per_token = 2 * layers * hidden_size * 2
    kv_cache_bytes = context_tokens * kv_bytes_per_token
    total_bytes = weights_bytes + kv_cache_bytes

    return {
        "effective_bits": effective_bits,
        "layers": layers,
        "hidden_size": hidden_size,
        "weights_bytes": weights_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_cache_bytes": kv_cache_bytes,
        "total_bytes": total_bytes,
    }


class LLMCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Memory Calculator")
        self.root.geometry("840x560")
        self.root.minsize(760, 500)

        self.model_size_var = tk.StringVar(value="7")
        self.quant_family_var = tk.StringVar(value="Q")
        self.bit_width_var = tk.StringVar(value="4")
        self.context_var = tk.StringVar(value="4096")
        self.status_var = tk.StringVar(value="Enter values to estimate memory usage.")
        self.quant_help_var = tk.StringVar(value=QUANTIZATION_HELP["Q"])

        self.result_vars = {
            "format": tk.StringVar(value="-"),
            "effective_bits": tk.StringVar(value="-"),
            "shape": tk.StringVar(value="-"),
            "weights": tk.StringVar(value="-"),
            "kv_per_token": tk.StringVar(value="-"),
            "kv_total": tk.StringVar(value="-"),
            "total": tk.StringVar(value="-"),
        }

        self._build_ui()
        self._bind_auto_calculation()
        self.calculate()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=16)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(2, weight=1)

        title = ttk.Label(
            main,
            text="LLM Memory Calculator",
            font=("Segoe UI", 18, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w")

        subtitle = ttk.Label(
            main,
            text=(
                "Estimate weight memory and KV cache from model size, quantization, "
                "bit width, and context window."
            ),
            wraplength=760,
        )
        subtitle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 16))

        input_frame = ttk.LabelFrame(main, text="Inputs", padding=16)
        input_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        result_frame = ttk.LabelFrame(main, text="Estimated Memory", padding=16)
        result_frame.grid(row=2, column=1, sticky="nsew")
        result_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Model parameter size (billions)").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self._combobox(
            input_frame,
            textvariable=self.model_size_var,
            values=MODEL_SIZE_OPTIONS,
            row=0,
        )

        ttk.Label(input_frame, text="Quantization type").grid(
            row=1, column=0, sticky="w", pady=(12, 6)
        )
        family_box = self._combobox(
            input_frame,
            textvariable=self.quant_family_var,
            values=("Q", "FP", "INT", "MXFP"),
            row=1,
            state="readonly",
        )
        family_box.bind("<<ComboboxSelected>>", self._on_family_changed)

        ttk.Label(
            input_frame,
            textvariable=self.quant_help_var,
            wraplength=300,
            foreground="#4f4f4f",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 4))

        ttk.Label(input_frame, text="Bit width").grid(
            row=3, column=0, sticky="w", pady=(12, 6)
        )
        self._combobox(
            input_frame,
            textvariable=self.bit_width_var,
            values=BIT_WIDTH_OPTIONS,
            row=3,
        )

        ttk.Label(input_frame, text="Context window (tokens)").grid(
            row=4, column=0, sticky="w", pady=(12, 6)
        )
        self._combobox(
            input_frame,
            textvariable=self.context_var,
            values=CONTEXT_OPTIONS,
            row=4,
        )

        button_row = ttk.Frame(input_frame)
        button_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        ttk.Button(button_row, text="Calculate", command=self.calculate).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(button_row, text="Reset", command=self.reset).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )

        rows = (
            ("Selected format", "format"),
            ("Effective bits / parameter", "effective_bits"),
            ("Estimated transformer shape", "shape"),
            ("Model weights", "weights"),
            ("KV cache / token", "kv_per_token"),
            ("KV cache total", "kv_total"),
            ("Total memory", "total"),
        )

        for row_index, (label_text, key) in enumerate(rows):
            ttk.Label(result_frame, text=label_text).grid(
                row=row_index, column=0, sticky="w", pady=6
            )
            ttk.Label(
                result_frame,
                textvariable=self.result_vars[key],
                font=("Segoe UI", 10, "bold"),
                wraplength=260,
            ).grid(row=row_index, column=1, sticky="e", pady=6)

        note = ttk.Label(
            result_frame,
            text=(
                "KV cache assumes fp16/bf16 storage for keys and values. Real usage depends "
                "on the actual model architecture and runtime."
            ),
            wraplength=320,
            foreground="#4f4f4f",
        )
        note.grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(14, 0))

        status = ttk.Label(
            main,
            textvariable=self.status_var,
            wraplength=760,
            foreground="#404040",
        )
        status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(14, 0))

    def _combobox(self, parent, textvariable, values, row, state="normal"):
        box = ttk.Combobox(parent, textvariable=textvariable, values=values, state=state)
        box.grid(row=row, column=1, sticky="ew", padx=(12, 0))
        return box

    def _bind_auto_calculation(self):
        for variable in (
            self.model_size_var,
            self.quant_family_var,
            self.bit_width_var,
            self.context_var,
        ):
            variable.trace_add("write", self._on_value_changed)

    def _on_value_changed(self, *_args):
        self.calculate()

    def _on_family_changed(self, _event=None):
        family = self.quant_family_var.get()
        self.quant_help_var.set(QUANTIZATION_HELP.get(family, ""))
        self.calculate()

    def reset(self):
        self.model_size_var.set("7")
        self.quant_family_var.set("Q")
        self.bit_width_var.set("4")
        self.context_var.set("4096")
        self.quant_help_var.set(QUANTIZATION_HELP["Q"])
        self.status_var.set("Values reset to a common small-model setup.")
        self.calculate()

    def calculate(self):
        try:
            params_billions = parse_positive_float(self.model_size_var.get())
            bits = parse_positive_int(self.bit_width_var.get())
            context_tokens = parse_positive_int(self.context_var.get())
            family = self.quant_family_var.get().strip().upper()

            if family not in QUANTIZATION_HELP:
                raise ValueError

            estimate = estimate_memory(params_billions, family, bits, context_tokens)
        except ValueError:
            self.status_var.set(
                "Use positive numbers for model size, bit width, and context window."
            )
            for variable in self.result_vars.values():
                variable.set("-")
            return

        self.quant_help_var.set(QUANTIZATION_HELP[family])

        self.result_vars["format"].set(f"{family}{bits}")
        self.result_vars["effective_bits"].set(f"{estimate['effective_bits']:.2f} bits")
        self.result_vars["shape"].set(
            f"{estimate['layers']} layers, hidden size ~{estimate['hidden_size']:,}"
        )
        self.result_vars["weights"].set(format_bytes(estimate["weights_bytes"]))
        self.result_vars["kv_per_token"].set(format_bytes(estimate["kv_bytes_per_token"]))
        self.result_vars["kv_total"].set(format_bytes(estimate["kv_cache_bytes"]))
        self.result_vars["total"].set(format_bytes(estimate["total_bytes"]))

        self.status_var.set(
            f"Estimate ready for a {params_billions:g}B model using {family}{bits} "
            f"with a {context_tokens:,}-token context window."
        )


def main():
    root = tk.Tk()
    app = LLMCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
