# Progetto Rete Siamese
## Passaggi Preliminari
- Installazione software necessario e preparazione ambiente
- rinominare `.bazelrc_linux` o `.bazelrc_windows` in `.bazelrc`
- rinominare, in ciascun `BUILD` file contenente un `py_binary`, tutte le variabili nell'`LD_LIBRARY_PATH` affinche' il prefisso della path
  alla repository corrisponda alla path effettiva in cui questa sia scaricata (*problema in corso di risoluzione*)
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

### `siamese_first` package
Implementazione di una rudimentale rete siamese seguendo gli appunti nel file `SiameseNotes.md` e le 
guide esempio keras su [Triplet Loss](https://keras.io/examples/vision/siamese_network/) e [Contrastive Loss](https://keras.io/examples/vision/siamese_contrastive/)

## Strumenti Utilizzati
- `python` e librerie specificate `requirements.in`
- `bazel`+`bazelisk` per il processo di building e deployment
- Vs Code come IDE
- `ruff` come strumento di linting e formatting (sia programme che estensione VSCode)

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

- esempio passare argomenti ad un `py_binary` fatto andare tramite bazel
```sh
bazel run //siamese_first:siamese_first -- --contrastive-loss --fast-train 
```

- VSCode ruff comandi: `Format Document`, `Organize Imports`, (non usare) `Fix All`

## Deployment Applicativo
Eseguire il comando
```sh
bazel build //:app_zip
```
nella convenience symlink `bazel-bin` viene creato un archivio `app_zip.zip` contenente tutto 
il necessario per eseguire l'applicazione con script nativi al sistema operativo che ha generato
lo zip.

Nota come, dato che alcuni dataset sono scaricati nel processo di building e quindi inclusi nel 
applicazione impacchettata, lo zip viene di 8.9 GB
(apparte MNIST, che invece \`e scaricato a runtime)

## Installazione in ambiente Windows
### Installazione software
#### Abilitare `Long Paths`
- `Win+R` ed eseguire `regedit`
- controllare nel registro `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem` e assicurarsi
  che la chiave `DWORD` chiamata `LongPathsEnabled` sia settata a `1`. Questo abilita la possibilita di abilitare
  Le long paths. Chiudere il registro
- `Win+R` ed eseguire `gpedit.msc` (se non e' disponibile, seguire [Questa Guida](https://www.majorgeeks.com/content/page/enable_group_policy_editor_in_windows_10_home_edition.html))
- Navigare in `Computer Configuration\Administrative Templates\System\Filesystem\` e assicurarsi che
  `Enable Win32 long paths` sia `Enabled`
- Sempre in `gpedit.msc`, navigare su `Computer Configuration\Windows Settings\Security Settings\Local Policies\User Right Assignment` 
  e assicurarsi che `Create symbolic links` contenga l'utente che esegue `bazel`

#### Installazione bazel
- Scaricare `chocolatey` eseguendo, da powershell come amministratore
  ```powershell
  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  ```
- Eseguire, da powershell come amministratore, `choco install bazelisk`
- Scaricare MSYS2 da [Questo Link](https://www.msys2.org/)
- Aggiungere nella variabile d'ambiente `BAZEL_SH` la path alla `usr\bin\bash.exe` di MSYS
- Aggiungere MSYS `bin` directory e bazel alla variabile d'ambiente `Path`
- avviare la console di msys e lanciare
  ```sh
  pacman -S zip unzip patch diffutils git
  ```
- Scaricare Visual Studio. Se e' scaricato in una directory diversa da quella default, ci sono variabili d'ambiente
  da settare come indicato nella [Guida d'installazione di Bazel](https://bazel.build/install/windows)
- Scaricare Java. [Link](https://www.oracle.com/java/technologies/downloads/#java11?er=221886)

#### Python e software opzionale
- Scaricare Python 3.10,
  ```powershell
  winget install Python.Python.3.10
  ```
  se installata con successo, questa versione dovrebbe essere visibile dal windows python launcher `py --list`

#### Software opzionale
- formatter: installazione `winget install astral-sh.ruff`, ed esecuzione `ruff format`

#### Testare se tutto funziona
```sh
bazel query @bazel_tools//src/conditions:all
```

### Costruzione Python Virtual Environment (per debugging)
```powershell
py -3.10 -m venv .venv
```
Questo comando crea un python virtual environment nella cartella `.venv`, al quale si puo "entrare" (cioe' settare
`Path` e `PYTHONPATH` environemnt variables in maniera appropriata) con il seguente comando (powershell)
```
.\.venv\Scripts\Activate.ps1
```
Per controllare la differenza delle variabili di ambiente eseguire
```powershell
Get-ChildItem Env:* | Sort-Object Name
```
In particolare, devono apparire le variabili di ambiente `VIRTUAL_ENV` e `VIRTUAL_ENV_PROMPT`

Possiamo dunque installare le dipendenze nel virtual environment con il comando
```
pip install -r requirements_windows.txt
```
La cartella `.venv\Lib\site-packages` dovrebbe contenere tutte le librerie installate

#### Integrazione Visual Studio Code
- Scaricare le estensioni `Python Extension Pack`, `Bazel`
- Assicurarsi che nel `settings.json`, dove sono definiti i terminali, sia inserita la path alla propria 
  installazione di Visual Studio.
- Definizione nel `launch.json` delle configurazioni di debugging utilizzanti il virtual environment
- Assicurarsi, con un file python `__main__.py` qualsiasi, che l'interprete Python usato per il debugging sia
  quello del virtual environment, dunque `Shift+Ctrl+P` e digitare `Python: Select Interpreter`, dunque
  selezionare il virtual environment
Nota che ogni volta che il virtual environment viene cambiato da VSCode, i terminali vanno riaperti, perche 
visual studio code inserisce i terminali nel venv in automatico. Verificare con
```powershell
Get-ChildItem Env:* | Sort-Object Name | findstr VIRTUAL_ENV
```

#### Testare che tensorflow sta usando la GPU
Nel Virtual Environment o da codice
```sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

# Packaging su windows
Il comando 
```
bazel build //:app_zip
```

Genera nella `bazel-bin` un `app_zip.zip` contenente tutti i files necessari per eseguire l'applicazione. Su windows, con `--build_python_zip=false`, la 
struttura delle directories dei runfiles generata e' incorretta. In particolare, in ogni cartella il cui nome termina con `runfiles`, e' contenuta soltanto la 
cartella `_main`.
Dovrebbe invece contenere anche tutte le cartelle il cui nome incomincia per `rules_python`, le quali finiscono in `_main` per via dell'implementazione di `pkg_zip`.
Due possibili soluzioni
1. Modificare il `.bazelrc` per utilizzare i python zips`--build_python_zip=true`, i quali rallentano lo sviluppo
2. Aggiustare manualmente la struttura delle directories nel `app_zip.zip`, dopo averne estratto il contenuto

Comando di esempio:
```powershell
PS > .\start.ps1 siamese_second -ArgumentList "--working-directory=$(Split-Path (Get-Location) -Parent)","--action=train"
```