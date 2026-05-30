# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""dataformat — the swm dataset format (register_format == register_terminals_format).

Union-distinct: a published clean-name alias over the one canonical implementation.
"""
from .swm_format import FORMAT_NAME, register_terminals_format

register_format = register_terminals_format

__all__ = ["FORMAT_NAME", "register_format", "register_terminals_format"]
