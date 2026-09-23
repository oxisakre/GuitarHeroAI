# Coordinación Claude ↔ Codex

Esta carpeta es un buzón entre agentes que trabajan sobre el mismo proyecto.
No es un mecanismo en tiempo real: cada agente debe leerla antes de empezar una
tarea y después de terminar una tanda de cambios.

Reglas:

1. Claude escribe sólo `CLAUDE_STATUS.md`; Codex escribe sólo `CODEX_STATUS.md`.
2. Antes de editar, cada agente declara los archivos que toma como propios.
3. No se edita un archivo que el otro marque `EN CURSO`.
4. Al terminar, se anotan pruebas, resultados, limitaciones y archivos liberados.
5. Los registros, modelos `.zip` y cambios del usuario no se borran ni revierten.

