"""Interfaz común de operador (`Operator`) y el guardián de memoria
(`MemoryBudget`) que usan los operadores bloqueantes.

`Operator` es la única abstracción que `pipeline.build_operator_tree`
necesita conocer: cualquier operador nuevo (por ejemplo un futuro
`PhysicalWindowFunction`) sólo tiene que implementar `output_schema` y
`execute` para poder combinarse con el resto sin tocar el motor — es el
punto de extensión que pide el DoD del proyecto.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .batch import RowBatch
from .errors import MemoryLimitExceededError
from .models import Schema

#: Límite de memoria por defecto para un operador bloqueante individual
#: (hash join, hash aggregate, sort) cuando no se especifica uno propio.
#: 256 MiB es un valor conservador para un proceso de demostración/test —
#: suficiente para varios cientos de miles de filas de ancho moderado sin
#: dejar que un operador crezca sin límite (regla D de `../CLAUDE.md`).
DEFAULT_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024


@runtime_checkable
class Operator(Protocol):
    """Nodo del árbol de ejecución ya construido a partir de un plan físico.

    `execute()` es un generador: cada operador debe tirar de sus hijos y
    producir sus propios `RowBatch` de salida de forma perezosa, para que
    un `LimitOperator` en la raíz pueda dejar de pedir batches en cuanto
    tiene suficientes filas sin forzar a ejecutar el resto del árbol.
    """

    @property
    def output_schema(self) -> Schema:
        """Esquema de los batches que produce `execute()` — conocido sin
        necesidad de ejecutar ningún dato, para que el operador padre pueda
        construir su propio esquema antes de arrancar la ejecución."""
        ...

    def execute(self) -> Iterator[RowBatch]:
        """Produce el resultado de este operador en batches."""
        ...


@dataclass
class MemoryBudget:
    """Contador de memoria acotado compartido por los operadores bloqueantes
    de un único árbol de ejecución.

    Estrategia de sincronización: esta clase no es seguro para acceso
    concurrente (no toma ningún lock) — un único árbol de ejecución
    (`pipeline.build_operator_tree`) se ejecuta secuencialmente dentro de un
    hilo, así que no hace falta; si en el futuro `nanosql` ejecuta planes en
    paralelo, cada ejecución debe recibir su propio `MemoryBudget` (ver
    `../CLAUDE.md`, regla D: nada de estado mutable compartido sin
    protección documentada).

    Este motor no implementa spill a disco: al superar `max_bytes`, un
    operador bloqueante falla explícitamente con `MemoryLimitExceededError`
    en vez de derramar filas a un fichero temporal — ver README, sección de
    arquitectura, para la justificación de este alcance.
    """

    max_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES
    _used_bytes: int = field(default=0, init=False)

    def reserve(self, operator_name: str, amount_bytes: int) -> None:
        if self._used_bytes + amount_bytes > self.max_bytes:
            raise MemoryLimitExceededError(
                operator_name, self._used_bytes + amount_bytes, self.max_bytes
            )
        self._used_bytes += amount_bytes

    def release(self, amount_bytes: int) -> None:
        self._used_bytes = max(0, self._used_bytes - amount_bytes)

    @property
    def used_bytes(self) -> int:
        return self._used_bytes


def estimate_batch_bytes(batch: RowBatch) -> int:
    """Estimación del tamaño en bytes de un `RowBatch` ya materializado.

    Exacta para columnas numéricas/booleanas (`np.ndarray.nbytes`);
    aproximada para columnas de texto (`dtype=object`), donde se suma
    `sys.getsizeof` de cada valor Python porque NumPy no reserva un tamaño
    fijo por celda para ellas — suficiente para hacer cumplir un límite de
    memoria razonable, no para contabilidad exacta de heap.
    """
    total = 0
    for name in batch.schema.columns:
        column = batch.columns[name]
        if column.dtype == object:
            total += sum(sys.getsizeof(value) for value in column)
        else:
            total += column.nbytes
        total += batch.nulls[name].nbytes
    return total
