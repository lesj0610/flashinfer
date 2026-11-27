"""ptxemu: sm_90a PTX to sm_80 PTX translator for the wgmma_emu emulation layer.

Modules: ``lexer`` (statement tokenizer), ``core`` (module model, rule driver, library merge;
PLAN section 5 API), ``check`` (inventory-based remaining-opcode check), ``cli``
(``python -m ptxemu``). Rule modules ``rules_wgmma``, ``rules_async`` and ``rules_misc`` are
loaded by ``core`` in that order.
"""
