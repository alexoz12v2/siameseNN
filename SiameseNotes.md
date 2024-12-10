# Teoria Reti Siamesi
## Intro
Una rete siamese usa *Similarity Learning*, ovvero cerca di imparare una funzione di similarit\`a 
tra due immagini.

Al fine di riconoscere una persona, le reti siamesi non prendono in input una singola immagine 
per ricondurla ad una delle "persone che la rete conosce" (essenzialmente una classe nota a priori), 
perch\`e costringerebbe un retraining ogni qual volta che si aggiunge una nuova persona 
da riconoscere, (e ti servono molte immagini per ogni persona).

La rete prende invece in input due immagini, 
- elabora ciascuna immagine attraverso due reti gemelle al fine di trasformare le immagini 
  in uno spazio delle features,
- utilizza una loss function comparativa per ottenere un indice di similarit\`a tra le due immagini
  e riconosci le due immagini come foto della stessa persona se tale misura \`e sufficientemente
  alta

## Note su Architettura
le reti gemelle hanno i *parametri condivisi*, quindi in fase di allenamento alleni effettivamente
una sola subnetwork

In fase di Tranining, si possono usare due possibili tipi di loss function
- Contrastive Loss Function
- Triplet Loss function

### Contrastive Loss
Prendi due immagini in input, falle passare attraverso le reti gemelle, calcolane gli embeddings, e
usa la seguente funzione di distanza tra le immagini come loss
```math
\mathcal{L}=\left(1-Y\right) \frac{1}{2}\left(D_W\right)^2 + (Y)\frac{1}{2}\left\{\mathrm{max}\left(0,m-D_W\right)\right\}^2
```
Dove 
- $D_W$ \`e la *Distanza Euclidea* tra i due vettori $Y$ e $\hat{Y}$
- $m$ \`e un *margine* positivo, iperparametro che detta la distanza minima che i due vettori devono 
  possedere per essere contati come diversi
- $Y$ valore vero del label

### Triplet Loss
Piuttosto che usare due immagini nel training, ne usiamo tre
- *Anchor*: immagine campionata dal dataset di training
- *Positive*: variazione della *Anchor* (data augmentation o foto diversa stessa persona)
- *Negative*: immagine diversa (altra persona)
```math
\mathcal{L}=\mathrm{max}\left(0, \mathrm{d}(a,p)-d(a,n)+m\right)
```
Dove
- $d$ funzione di distanza tra i due embeddings, eg Norma L2 o [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- $a$, $n$, $p$ rispettivamente embeddings di Anchor, Negative, Positive
- $m$ margine settato a priori

## Nota a margine
- se non usi i blocchi residuali, ma gli inception modules, allora ci sono dei rami di output secondario
  usati in training per iniettare gradiente aggiuntivo, e prende output intermedio da entrambe le reti 
  gemelle, la "Correlation Loss". Vedi [Immagine Sito](https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513)