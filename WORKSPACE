# Download datasets cats and dogs
# generera nella cartella ~/.cache/bazel/_bazel_{USER}/{WORKSPACE dir hash}/execroot/{workspace name}/external/cats_and_dogs (essenzialmente la cartella del bazel sandbox)
# un WORKSPACE + una cartella "file" che contiene lo zip
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
http_file(
    name = "cats_and_dogs",
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip",
)

# dataset per siamese networks: left => dataset anchors, right => dataset positive per ogni anchor
# sto usando due http_file targets, che creano un filegroup di un solo zip, piuttosto che una sola
# http_file che crea un filegroup di 2 zips, perche la mia rule custom zip_extract gestisce labels
# che si riferiscono ad un filegroup con un solo file
# quello che http_file fa e' una GET all'url che gli hai specificato
http_file(
    name = "siamese_left",
    url = "https://drive.usercontent.google.com/download?id=1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34&authuser=0&confirm=t&uuid=9cf14af2-9021-4aa1-ae66-707c34a5b2c7&at=APvzH3r0KuHiI11CxX78ZPJk8pB1:1733746276346",
)

http_file(
    name = "siamese_right",
    url = "https://drive.usercontent.google.com/download?id=1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW&authuser=0&confirm=t&uuid=0a26a7b6-8d05-4d19-ba60-112b2db37dee&at=APvzH3qj0KXtwq86j3DM9fiWyb-7:1733746709914",
)
