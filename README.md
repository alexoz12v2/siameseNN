# Progetto Rete Siamese


## Procedura di generazione Build Files e workflow con Visual Studio Code
Si suppone che in vscode siano installate le estensioni `Python Extension Pack` e `Bazel`.
1. Per prima cosa, devono essere generate versioni e hash delle dipendenze specificate nel file 
   `requirements.in`, dunque eseguire il comando 
   ```sh
    bazel run //:requirmements.update
   ```
   se presenta errore "non riesco a trovare i files `requirements_lock.txt` e 
   `requirements_windows.txt`", creare dei files vuoti

2. In quanto i packages python possono essere rinominati quando sono importati, una lista di dipendenze 
   non basta, bensi deve essere costruito un grafo di dipendenze che tenga conto di eventuali redirezioni
   e renaming fatti dai python packages. Dunque eseguire il comando
   ```sh
   bazel run //:gazelle_python_manifest.update
   ```
   se si ripresenta il problema a step 1, creare file vuoto `gazelle_python.yaml`

3. Gazelle \`e capace di generare files di build relativi a eventuali files nominati `__init__.py` per 
   `py_library`, `__main__.py` per `py_binary`, e `__test__.py` per `py_test` nei vari folders marcati 
   come bazel packages con il file `BUILD`. Dunque eseguire il comando
   ```sh
   bazel run //:gazelle
   ```
   Preferisco specificare le `py_library` e quant'altro manualmente, ma eseguire il comando per 
   sicurezza

4. Creare un virtual environment per python il quale conterr\`a la versione specificata di python e le 
   dipendenze calcolate con gazelle, e permetter\`a a vs code di poter usare test e runner per 
   debugging, oltre a poter navigare a definizione. Per la creazione del virtual environment, 
   grazie alle rules di aspect build, per ciascuna rule python viene dichiarato un rule target
   del tipo `<label_target>.venv`. Dunque eseguendo quel target verr\`a creato nel workspace la cartella
   con symlinks e dipendenze necessarie per far andare i python scripts in sviluppo, e costruire 
   l'eseguibile nativo che fa il packaging dell'ambiente, dipendenze, e interprete python. 
   Per la creazione del venv, eseguire il comando sopra specificato, esempio
   ```sh
   bazel run //app:keras_test.venv
   ```

5. Una volta creato il venv, navigare in un file python che fa parte del target per il quale hai
   creato il venv, e in basso a sinistra, a sinistra di "Spaces, LF", ci sar\`a il nome dell'interprete
   python usato. Se lo clicchi, vedrai che \`e disponibile in venv appena creato. Selezionalo.
   Adesso ogni volta che provi a fare run e debug verr\`a usato il venv.

Caveats:
- Se usi un filesystem che non supporta symbolick links, come `exfat` (provato anche con 
  l'`exfat-fuse` driver), hai problemi perch\`e bazel usa molto i link simbolici, esempio per generare 
  dei puntatori di convenienza dalla directory del workspace alla sandbox bazel in cui ci sono gli 
  artefatti generati
