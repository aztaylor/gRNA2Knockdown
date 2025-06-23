
#set page(margin: 1.00in)
#set par(leading: 0.55em, spacing: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)

= Genetic Control in _E.coli_ Leveraging Auto-encorder Based cRNA Design and CRISPR-#emph[rfx]Cas13d.

== Abstract
== Introduction
   1. Type VI class 2 CRISPR associated proteins (Cas) expand the Clustered Regularly Interspaced Short Palindromic Repeats (CRISPR) toolbox into the transcriptome #super[@abudayyeh2017].
   2. Previouly, efforts in RNA-guided RNA-targeting CRISPR-Cas systems, such as Cas13d, have been focused on their ability to modify, detect, and degrade various RNA species in eukarytic cells.
   3. RNA-guided RNA-targeting CRISPR-Cas systems allow for precise control of protein expression, whithout constraints of posed with DNA-targeting CRISPR-cas systems. $mu +sigma =Delta$
   4. In this study,we explore dCas13d in _E.coli_ for genetic control, leveraging the design of custom RNA guides (cRNA) using an predictive neural network approach.
== Results
  1. *Deactivated Cas13d achieves various degress of expression control in plasmid based systems.*
    1. We developed a library of tiled crRNA guides spanning the entirity of the bicistronic element design (BCD) 2 - GFP reporter cassestte @figure1.
    \
    #figure(image("../../Figures/paperFig1v2_Mech_and_plasmids.png", width: 100%), caption: "Mechanism of dCas13d and plasmid design for GFP reporter system") <figure1>
    \
    2. The dCas13d system was able to achieve a range of expression control, from -200% to  ~10% of the GFP reporter @figure2.
    \
    #figure(image("../../Figures/80memberLibraryDifferenceBarChart_2_0_3.png", width: 100%), caption: "Mechanism of dCas13d and plasmid design for GFP reporter system")<figure2>
    \
  2. *Neural networks can predict dCas13d-gRNA effects based on sequence.*
    1. To predict the effect a specific gRNA would have on the GFP expression plasmid, we developed a two stage neural network consisting of an autoencoder and a convolutional neural network (CNN).
    2. The network was able to embed the gRNA sequence into an 18 node latent space and predict the effect of the gRNA on GFP expression wipaperFigIIIvBlurry.png reconstruction.
    \
    #figure(image("../../Figures/paperFigIIIvBlurry.png", width: 100%), caption: "Mechanism of dCas13d and plasmid design for GFP reporter system")<figure3>
    \
  3. To determine the generalizability of dCasRx, we tested it ability to target chromosomal housekeeper genes, specifically _gryA_ and _gyrB_.
    1. Due to its'role in DNA torsional force maintanence, we hypostesized a deficit of either subunit would result in a growth defect.
    2. Comparing growth the dCasRx-#emph[gyrA] and dCasRx-#emph[gyrB] strains to a control strain, we observed a significant increase in stationary cell density in the dCasRx-#emph[gyrA] strain, but not in the dCasRx-#emph[gyrB] strain @figure4.
    \
    #figure(image("../../Figures/dCas9_CasRxComp.png", width: 100%), caption: "Mechanism of dCas13d and plasmid design for GFP reporter system")<figure4>
    \
  4. Cells containing gyrase targeting _gyrA_ had significantly different gene epxression profiles than those treated with the Gyrase inhibitor Norfloxacin.
    1. We observed that while chemically treated cells tended to respond to norfloxacin with upregualtion of DNA repair genes, dCasRx tended
    2. This suggests that the dCasRx system can be used to modulate gene expression in a manner similar to chemical inhibitors, but with greater precision and control.
== Discussion
   1. Implications of findings
   2. Comparison with existing methods
   3. Limitations and future work

== Conclusion
#bibliography("ATCitations.bib", title:"References", style:"nature", full: true)