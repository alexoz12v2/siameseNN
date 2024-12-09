# Progetto Rete Siamese
## Bazel Packages
### `app` package
Modello MLP semplice, che con 100 epoche arriva al 10% di accuracy con un dataset sintetizzato.
Non \`e il massimo, ma \`e un inizio

### `classification_from_scratch` package
Modello convolutivo utilizzante grouped convolutions e residual blocks, seguito da un layer fully 
connected per fare classificazione binaria del dataset "cats and dogs"
Il dataset viene scaricato come `.zip` dalla bazel repository rule `http_file` nel `WORKSPACE` file, 
il quale inserisce lo zip scaricato in un nuovo `WORKSPACE` generato in 
`{sandbox}/execroot/{nome mio workspace}/external/{nome target http_file}`, esempio
```sh
/home/alessio/.cache/bazel/_bazel_alessio/3f053d8d1584724eb2a7dbb5879cd649/execroot/build_file_generation_example/external/cats_and_dogs
- file/
  - downloaded.zip <- cats and dogs zip
- WORKSPACE
```
Dopodiche lo zip viene estratta da una rule custom in `bazel/zip_extract.bzl` che crea un depset
dal risultato dell'unzip, i cui files sono inseriti in una directory relativa al package corrente
specificata in input alla rule.

Il codice in se invece, segue la spiegazione fornita dalla [documentazione keras](https://keras.io/examples/vision/image_classification_from_scratch/)

## Strumenti Utilizzati
- `python` e librerie specificate `requirements.in`
- `bazel`+`bazelisk` per il processo di building e deployment
- Vs Code come IDE
- `ruff` come strumento di linting e formatting

Tutti i dataset utilizzati possono essere scaricati in due modi
- Tramite bazel, repository rule `http_file` -> Integrate nello zip deployato (e sono directories 
  read only con al loro interno files read only)
- Tramite python -> dunque scaricate a runtime

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
   sicurezza.
   - gazelle in automatico conta anche le dipendenze interne come dipendenze da risolvere nel
     `requirements.txt`, il che non \`e esatto. Dunque, in un qualsiasi `BUILD` file, meglio se fatto
     dopo la dichiarazione del target python, inserire `gazelle:resolve py {DEP} {TARGET_LABEL}`. Esempio
     ```sh
     py_binary(...)
     # gazelle:resolve py utils //app/utils:layers
     ```
      Ma questo succede perch\`e non hai espresso la `import` nel file python con una path relativa al `WORKSPACE`,
      se correggi questo l'errore dovrebbe andare via.
      Infine, al fine di far si che bazel e gazelle non scannerizzino la directory che contiene il virtual 
      environment, possiamo inserire nel `.bazelignore` una riga del tipo `.{package}+{target}.venv`. Eg:
      ```sh
      .app+keras_test.venv
      ```
   - Inoltre, le cartelle dei virtual environment devono essere anche aggiunte nel `.ruff.toml`, al
     fine di poter usare il formatter e linter `ruff` soltanto nei files che ho effettivamente scritto
     Nota che 
   - Quando aggiungi nelle `deps` di un `py_binary` una dipendenza che non usi direttamente ma che ti
     serve indirettamente, (esempio `pyqt6` affinch\`e `matplotlib` sia capace di aprire finestre) ,
     noti che gazelle rimuove le dipendenze non usate. Quindi in uno dei files del python target devi
     inserire un import che non usi (`ruff check` infatti si lamenta)

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

Piuttosto che eseguire un file python dal virtual environment, il quale ad esempio, 
\`e equivalente al comando
```sh
"../siameseNN/.classification_from_scratch+classification_from_scratch.venv/bin/python" "/siameseNN/classification_from_scratch/__main__.py"
```
il che permette a visual studio code di attaccare il suo debugger.
Per eseguire invece il packaged executable creato da bazel, comando
```sh
bazel run //classification_from_scratch:classification_from_scratch
```

Per formattare in autoamtico il file, eseguire il comando, dalla directory della repo
```sh
ruff format
```
Se non hai `ruff`, scaricarlo seguendo la [guida](https://github.com/astral-sh/ruff)

Caveats:
- Se usi un filesystem che non supporta symbolick links, come `exfat` (provato anche con 
  l'`exfat-fuse` driver), hai problemi perch\`e bazel usa molto i link simbolici, esempio per generare 
  dei puntatori di convenienza dalla directory del workspace alla sandbox bazel in cui ci sono gli 
  artefatti generati

- eseguire un `__main__.py` da bazel o da python virtual environment ha una grossa differenza: la 
  current working directory. D\`a problemi nella lettura del dataset, che tramite bazel \`e scaricato
  nella directory del sandbox
  Una possibile fix specifica per visual studio code pu\`o esseere la creazione di 
  un `.vscode/launch.json`, che specifica come `"cwd"` la 
  `${workspaceFolder}/bazel-bin/${relativeFileDirname}/${fileDirnameBasename}.runfiles/{nome workspace}` convenience
  symlink che bazel genera.
  Per comodit\`a, il file `.vscode/launch.json` \`e stato incluso
  In particolare, `bazel run` esegue i propri target con current working directory pari ai 
  [runfiles](https://stackoverflow.com/questions/70256713/bazel-c-project-how-to-specify-working-directory-for-run-command) del target.
  *Attenzione*: Questo significa che vscode riesce ad eseguire un binario il cui target name coincide con
  la directory del bazel package, altrimenti devi aggiungere una configurazione ad hoc
  Riferimento [VS Code](https://code.visualstudio.com/docs/editor/debugging#_launch-configurations)
  Nota che ogni qual volta fai debugging, il cambio di working directory rimane nel terminale con il 
  virtual environment

### Note personali

- Comando `kill -KILL %1` per terminare il processo sospeso con CTRL+z

- vedere meglio la libreria `pillow`, che fa un lavoro migliore nel filtrare le immagini corrotte
  (non funziona sempre comunque)
  
- Quando editi i files python, attenzione a editare le sorgenti e non quelli nella bazel sandbox