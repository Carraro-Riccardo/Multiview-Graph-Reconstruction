@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Pipeline completa: ricostruzione, valutazione e visualizzazione.
rem Uso:
rem   run_pipeline.cmd          tutti i nodi
rem   run_pipeline.cmd 2 7 9    solo i nodi indicati

cd /d "%~dp0.."

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=python"
    ) else (
        echo ERRORE: Python non trovato. Cercati "py" e "python". 1>&2
        exit /b 127
    )
)

echo.
echo === 1/3 Ricostruzione ===
%PYTHON% run.py %*
if errorlevel 1 goto :failed

echo.
echo === 2/3 Valutazione e confronto con gli standard ===
%PYTHON% evaluate.py %*
if errorlevel 1 goto :failed

echo.
echo === 3/3 Visualizzazione ===
if not "%~1"=="" (
    %PYTHON% visualize.py %*
    if errorlevel 1 goto :failed
) else (
    %PYTHON% visualize.py --all
    if errorlevel 1 goto :failed
)

echo.
echo === PIPELINE COMPLETATA ===
echo Tabella similarita: %CD%\output\standard_comparison\similarity_table.html
echo Best match CSV:     %CD%\output\standard_comparison\best_matches.csv
echo Confronto grafico:  %CD%\output\standard_comparison\report_completo.html
echo Grafici per nodo:   %CD%\output\multiview\node*\fused_graph_plot.png
exit /b 0

:failed
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo ERRORE: pipeline interrotta con exit code %EXIT_CODE%. 1>&2
exit /b %EXIT_CODE%
