from dmpfold import aln_to_coords

# Default options
coords, confs = aln_to_coords("PF10963.aln")

# Change options
coords, confs = aln_to_coords("PF10963.aln", device="cuda", template="template.pdb", iterations=10, minsteps=100)