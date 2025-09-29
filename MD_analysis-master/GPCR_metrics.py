#!/usr/bin/env python3
"""
gpcr activation analysis for gnrh1r compares inactive and active states
"""

import numpy as np
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')


class GPCRAnalyzer:
    
    def __init__(self):
        # key residue positions for gnrh1r
        self.res = {
            'TM3_center': 139, 'TM6_center': 282, 'TM7_center': 320,
            'DPxxY_D': 319, 'DPxxY_P1': 320, 'DPxxY_Y': 323,
            'CWxP_C': 279, 'CWxP_W': 280, 'CWxP_P': 282,
            'ICL2_start': 144, 'ICL2_end': 153,
            'ICL3_start': 243, 'ICL3_end': 260,
        }
        self.results = {}
    
    def get_ca(self, struct, chain, resnum):
        try:
            for model in struct:
                for ch in model:
                    if ch.id == chain:
                        for residue in ch:
                            if residue.id[1] == resnum and 'CA' in residue:
                                return residue['CA'].coord
        except:
            pass
        return None
    
    def get_residue(self, struct, chain, resnum):
        try:
            for model in struct:
                for ch in model:
                    if ch.id == chain:
                        for residue in ch:
                            if residue.id[1] == resnum:
                                return residue
        except:
            pass
        return None
    
    def calc_dist(self, pos1, pos2):
        if pos1 is not None and pos2 is not None:
            return np.linalg.norm(pos1 - pos2)
        return None
    
    def get_helix_vec(self, struct, chain, start, end):
        pos1 = self.get_ca(struct, chain, start)
        pos2 = self.get_ca(struct, chain, end)
        if pos1 is not None and pos2 is not None:
            vec = pos2 - pos1
            return vec / np.linalg.norm(vec)
        return None
    
    def calc_angle(self, vec1, vec2):
        if vec1 is not None and vec2 is not None:
            cos_ang = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
            return np.degrees(np.arccos(cos_ang))
        return None
    
    def analyze_tm_distances(self, inactive, active, ichain, achain):
        results = {}
        pairs = [('TM3-TM6', 'TM3_center', 'TM6_center'),
                 ('TM3-TM7', 'TM3_center', 'TM7_center'),
                 ('TM6-TM7', 'TM6_center', 'TM7_center')]
        
        for name, r1, r2 in pairs:
            ipos1 = self.get_ca(inactive, ichain, self.res[r1])
            ipos2 = self.get_ca(inactive, ichain, self.res[r2])
            apos1 = self.get_ca(active, achain, self.res[r1])
            apos2 = self.get_ca(active, achain, self.res[r2])
            
            idist = self.calc_dist(ipos1, ipos2)
            adist = self.calc_dist(apos1, apos2)
            
            if idist and adist:
                results[name] = {'inactive': idist, 'active': adist, 'change': adist - idist}
        return results
    
    def analyze_tm_rotations(self, inactive, active, ichain, achain):
        results = {}
        helices = [('TM3', 'TM3_center'), ('TM6', 'TM6_center'), ('TM7', 'TM7_center')]
        
        for helix, center_key in helices:
            center = self.res[center_key]
            ivec = self.get_helix_vec(inactive, ichain, center-15, center+15)
            avec = self.get_helix_vec(active, achain, center-15, center+15)
            angle = self.calc_angle(ivec, avec)
            if angle is not None:
                results[helix] = angle
        return results
    
    def analyze_dpxxy(self, inactive, active, ichain, achain):
        results = {}
        dpxxy_res = [self.res['DPxxY_D'], self.res['DPxxY_P1'], self.res['DPxxY_Y']]
        
        ipos = [self.get_ca(inactive, ichain, r) for r in dpxxy_res]
        apos = [self.get_ca(active, achain, r) for r in dpxxy_res]
        ipos = [p for p in ipos if p is not None]
        apos = [p for p in apos if p is not None]
        
        if len(ipos) == 3 and len(apos) == 3:
            results['center'] = np.linalg.norm(np.mean(apos, axis=0) - np.mean(ipos, axis=0))
        
        # asp carboxyl
        iasp = self.get_residue(inactive, ichain, self.res['DPxxY_D'])
        aasp = self.get_residue(active, achain, self.res['DPxxY_D'])
        if iasp and aasp and 'OD1' in iasp and 'OD2' in iasp and 'OD1' in aasp and 'OD2' in aasp:
            icarb = (iasp['OD1'].coord + iasp['OD2'].coord) / 2
            acarb = (aasp['OD1'].coord + aasp['OD2'].coord) / 2
            results['asp'] = np.linalg.norm(acarb - icarb)
        
        # tyr ring
        ityr = self.get_residue(inactive, ichain, self.res['DPxxY_Y'])
        atyr = self.get_residue(active, achain, self.res['DPxxY_Y'])
        if ityr and atyr:
            ring_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
            iring = [ityr[a].coord for a in ring_atoms if a in ityr and a in atyr]
            aring = [atyr[a].coord for a in ring_atoms if a in ityr and a in atyr]
            if len(iring) >= 4:
                results['tyr'] = np.linalg.norm(np.mean(aring, axis=0) - np.mean(iring, axis=0))
        return results
    
    def analyze_cwxp(self, inactive, active, ichain, achain):
        results = {}
        
        # trp indole
        itrp = self.get_residue(inactive, ichain, self.res['CWxP_W'])
        atrp = self.get_residue(active, achain, self.res['CWxP_W'])
        if itrp and atrp:
            indole_atoms = ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']
            iindole = [itrp[a].coord for a in indole_atoms if a in itrp and a in atrp]
            aindole = [atrp[a].coord for a in indole_atoms if a in itrp and a in atrp]
            if len(iindole) >= 6:
                results['trp'] = np.linalg.norm(np.mean(aindole, axis=0) - np.mean(iindole, axis=0))
                
                # ring rotation
                def get_normal(coords):
                    center = np.mean(coords, axis=0)
                    v1, v2 = coords[0] - center, coords[1] - center
                    normal = np.cross(v1, v2)
                    mag = np.linalg.norm(normal)
                    return normal / mag if mag > 0 else None
                
                inorm = get_normal(iindole)
                anorm = get_normal(aindole)
                if inorm is not None and anorm is not None:
                    results['trp_rotation'] = np.degrees(np.arccos(np.clip(np.dot(inorm, anorm), -1, 1)))
        
        # cys sulfur
        icys = self.get_residue(inactive, ichain, self.res['CWxP_C'])
        acys = self.get_residue(active, achain, self.res['CWxP_C'])
        if icys and acys and 'SG' in icys and 'SG' in acys:
            results['cys'] = np.linalg.norm(acys['SG'].coord - icys['SG'].coord)
        return results
    
    def analyze_loops(self, inactive, active, ichain, achain):
        results = {}
        loops = {'ICL2': (self.res['ICL2_start'], self.res['ICL2_end']),
                 'ICL3': (self.res['ICL3_start'], self.res['ICL3_end'])}
        
        for name, (start, end) in loops.items():
            ipos = [self.get_ca(inactive, ichain, r) for r in range(start, end+1)]
            apos = [self.get_ca(active, achain, r) for r in range(start, end+1)]
            ipos = [p for p in ipos if p is not None]
            apos = [p for p in apos if p is not None]
            if ipos and apos:
                results[name] = np.linalg.norm(np.mean(apos, axis=0) - np.mean(ipos, axis=0))
        return results
    
    def analyze_tm6(self, inactive, active, ichain, achain):
        results = {}
        ipos = self.get_ca(inactive, ichain, self.res['TM6_center'])
        apos = self.get_ca(active, achain, self.res['TM6_center'])
        if ipos is not None and apos is not None:
            disp = apos - ipos
            results['lateral'] = np.sqrt(disp[0]**2 + disp[1]**2)
            results['vertical'] = abs(disp[2])
        return results
    
    def run(self, inactive_pdb, active_pdb, ichain='A', achain='P'):
        parser = PDBParser(QUIET=True)
        
        try:
            print("loading structures...")
            inactive = parser.get_structure('inactive', inactive_pdb)
            active = parser.get_structure('active', active_pdb)
            
            print("analyzing tm distances...")
            self.results['tm_distances'] = self.analyze_tm_distances(inactive, active, ichain, achain)
            
            print("analyzing helix rotations...")
            self.results['tm_rotations'] = self.analyze_tm_rotations(inactive, active, ichain, achain)
            
            print("analyzing dpxxy motif...")
            self.results['dpxxy'] = self.analyze_dpxxy(inactive, active, ichain, achain)
            
            print("analyzing cwxp motif...")
            self.results['cwxp'] = self.analyze_cwxp(inactive, active, ichain, achain)
            
            print("analyzing loops...")
            self.results['loops'] = self.analyze_loops(inactive, active, ichain, achain)
            
            print("analyzing tm6...")
            self.results['tm6'] = self.analyze_tm6(inactive, active, ichain, achain)
            
            print("\nanalysis complete\n")
            return self.results
            
        except Exception as e:
            print(f"error: {e}")
            return None
    
    def print_results(self):
        if not self.results:
            print("no results available")
            return
        
        print("="*70)
        print("GPCR ACTIVATION ANALYSIS")
        print("="*70)
        
        if 'tm_distances' in self.results:
            print("\nTransmembrane Distance Changes:")
            for pair, data in self.results['tm_distances'].items():
                print(f"  {pair:10s}: {data['inactive']:.2f} -> {data['active']:.2f} A (change: {data['change']:+.2f} A)")
        
        if 'tm_rotations' in self.results:
            print("\nHelix Rotations:")
            for helix, angle in self.results['tm_rotations'].items():
                print(f"  {helix}: {angle:.1f} degrees")
        
        if 'dpxxy' in self.results:
            print("\nDPxxY Motif:")
            for key, val in self.results['dpxxy'].items():
                print(f"  {key}: {val:.2f} A")
        
        if 'cwxp' in self.results:
            print("\nCWxP Motif:")
            for key, val in self.results['cwxp'].items():
                unit = ' degrees' if 'rotation' in key else ' A'
                print(f"  {key}: {val:.2f}{unit}")
        
        if 'loops' in self.results:
            print("\nIntracellular Loops:")
            for loop, disp in self.results['loops'].items():
                print(f"  {loop}: {disp:.2f} A")
        
        if 'tm6' in self.results:
            print("\nTM6 Movement:")
            for key, val in self.results['tm6'].items():
                print(f"  {key}: {val:.2f} A")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    inactive_pdb = "/users/nkb19202/angles_distance/inactive.pdb"
    active_pdb = "/users/nkb19202/angles_distance/active.pdb"
    
    analyzer = GPCRAnalyzer()
    analyzer.run(inactive_pdb, active_pdb, ichain='A', achain='P')
    analyzer.print_results()
