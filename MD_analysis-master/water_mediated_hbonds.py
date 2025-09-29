#!/usr/bin/env python3
"""
water-mediated interaction analysis for the Gnrh1r-gnrh system
"""

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
import pickle
import time


class WaterAnalyzer:
    
    def __init__(self, water_cutoff=3.5, direct_cutoff=3.5, frame_step=20):
        self.water_cutoff = water_cutoff
        self.direct_cutoff = direct_cutoff
        self.frame_step = frame_step
        self.time_step = 0.02  # ns per frame
        
        # target residues with atoms
        self.targets = {
            "N87": {"segment": "PROA", "resid": 87, "atoms": ["OD1", "ND2"]},
            "E90": {"segment": "PROA", "resid": 90, "atoms": ["OE1", "OE2"]},
            "D319": {"segment": "PROA", "resid": 319, "atoms": ["OD1", "OD2"]},
            "Y323": {"segment": "PROA", "resid": 323, "atoms": ["OH"]},
            "R8_GnRH": {"segment": "PROB", "resid": 8, "atoms": ["NH1", "NH2", "NE"]}
        }
        
        self.results = {}
    
    def load_trajectories(self, traj_files):
        """load and concatenate trajectories"""
        print("loading trajectories...")
        
        parts = []
        labels = []
        boundaries = [0]
        
        for label, (traj_file, top_file) in sorted(traj_files.items()):
            print(f"  loading {label}...")
            traj = md.load(traj_file, top=top_file)
            parts.append(traj)
            labels.append(label)
            boundaries.append(boundaries[-1] + traj.n_frames)
            print(f"    {traj.n_frames} frames")
        
        combined = md.join(parts)
        print(f"  total: {combined.n_frames} frames")
        
        return combined, boundaries, labels
    
    def find_atoms(self, traj):
        """find water and target atoms"""
        print("finding atoms...")
        
        # all water oxygens
        waters = traj.topology.select("resname HOH and name O")
        print(f"  found {len(waters)} water molecules")
        
        # target atoms
        target_atoms = {}
        for name, info in self.targets.items():
            atom_names = " or name ".join(info['atoms'])
            sel = f"segname {info['segment']} and resSeq {info['resid']} and (name {atom_names})"
            atoms = traj.topology.select(sel)
            
            if len(atoms) > 0:
                target_atoms[name] = atoms
                atom_list = [traj.topology.atom(a).name for a in atoms]
                print(f"  {name}: {len(atoms)} atoms - {atom_list}")
        
        return waters, target_atoms
    
    def calc_distances(self, traj, target_atoms, frames):
        """calculate pairwise distances"""
        print("\ncalculating distances...")
        
        results = {}
        pairs = [("R8_GnRH", "E90"), ("R8_GnRH", "N87"), ("E90", "N87"),
                 ("N87", "D319"), ("N87", "Y323"), ("D319", "Y323")]
        
        for res1, res2 in pairs:
            if res1 not in target_atoms or res2 not in target_atoms:
                continue
            
            print(f"  {res1} - {res2}...")
            atoms1 = target_atoms[res1]
            atoms2 = target_atoms[res2]
            min_dists = []
            
            for idx in frames:
                coords = traj.xyz[idx] * 10  # nm to angstrom
                dmat = cdist(coords[atoms1], coords[atoms2])
                min_dists.append(np.min(dmat))
            
            times = np.arange(len(min_dists)) * self.frame_step * self.time_step
            results[f"{res1}-{res2}"] = {
                'distances': np.array(min_dists),
                'time': times
            }
        
        return results
    
    def calc_rdf(self, traj, target_atoms, waters, frames):
        """calculate radial distribution functions"""
        print("\ncalculating rdf...")
        
        results = {}
        residues = ["R8_GnRH", "E90", "N87"]
        max_dist = 10.0
        n_bins = 50
        
        for name in residues:
            if name not in target_atoms:
                continue
            
            print(f"  {name}...")
            all_dists = []
            
            for idx in frames[::2]:  # use fewer frames for rdf
                coords = traj.xyz[idx] * 10
                res_coords = coords[target_atoms[name]]
                water_coords = coords[waters]
                
                for rc in res_coords:
                    dists = np.linalg.norm(water_coords - rc, axis=1)
                    all_dists.extend(dists[dists <= max_dist])
            
            if len(all_dists) == 0:
                continue
            
            all_dists = np.array(all_dists)
            hist, edges = np.histogram(all_dists, bins=n_bins, range=(0, max_dist))
            centers = (edges[:-1] + edges[1:]) / 2
            width = edges[1] - edges[0]
            
            # normalise by shell volume
            volumes = 4 * np.pi * centers**2 * width
            volumes[volumes == 0] = 1
            rdf = hist / volumes
            
            # normalise to bulk density
            if np.max(rdf) > 0:
                rdf = rdf / np.mean(rdf[centers > 8])
            
            results[f"{name}-water"] = {
                'r': centers,
                'gr': rdf,
                'n_distances': len(all_dists)
            }
        
        return results
    
    def calc_interactions(self, traj, target_atoms, waters, frames):
        """classify interaction types"""
        print("\nanalyzing interactions...")
        
        results = {}
        pairs = [("R8_GnRH", "E90"), ("R8_GnRH", "N87"), ("E90", "N87")]
        
        for res1, res2 in pairs:
            if res1 not in target_atoms or res2 not in target_atoms:
                continue
            
            print(f"  {res1} - {res2}...")
            atoms1 = target_atoms[res1]
            atoms2 = target_atoms[res2]
            
            direct = 0
            water_med = 0
            no_contact = 0
            
            for idx in frames:
                coords = traj.xyz[idx] * 10
                coords1 = coords[atoms1]
                coords2 = coords[atoms2]
                water_coords = coords[waters]
                
                # check direct contact
                min_dist = np.min(cdist(coords1, coords2))
                
                if min_dist <= self.direct_cutoff:
                    direct += 1
                else:
                    # check water bridge
                    d1 = cdist(water_coords, coords1).min(axis=1)
                    d2 = cdist(water_coords, coords2).min(axis=1)
                    
                    if np.any((d1 <= self.water_cutoff) & (d2 <= self.water_cutoff)):
                        water_med += 1
                    else:
                        no_contact += 1
            
            total = len(frames)
            results[f"{res1}-{res2}"] = {
                'direct': direct / total,
                'water': water_med / total,
                'none': no_contact / total
            }
            
            print(f"    direct: {direct/total*100:.1f}%, water: {water_med/total*100:.1f}%, none: {no_contact/total*100:.1f}%")
        
        return results
    
    def calc_bridges(self, traj, target_atoms, waters, frames):
        """calculate water bridge occupancies"""
        print("\ncalculating water bridges...")
        
        results = {}
        names = list(target_atoms.keys())
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
        
        for res1, res2 in pairs:
            count = 0
            
            for idx in frames:
                coords = traj.xyz[idx] * 10
                coords1 = coords[target_atoms[res1]]
                coords2 = coords[target_atoms[res2]]
                water_coords = coords[waters]
                
                d1 = cdist(water_coords, coords1).min(axis=1)
                d2 = cdist(water_coords, coords2).min(axis=1)
                
                if np.any((d1 <= self.water_cutoff) & (d2 <= self.water_cutoff)):
                    count += 1
            
            occ = count / len(frames)
            results[f"{res1}--{res2}"] = {'occupancy': occ}
            print(f"  {res1}--{res2}: {occ:.3f}")
        
        return results
    
    def calc_legacy_bridges(self, traj, target_atoms, waters, frames, boundaries, labels):
        """legacy bridge calculation for comparison"""
        print("\ncalculating legacy bridges...")
        
        # rename for legacy format
        legacy = {}
        mapping = {
            "N87": "PROA_87", "E90": "PROA_90", "D319": "PROA_319",
            "Y323": "PROA_323", "R8_GnRH": "PROB_8"
        }
        
        for new, old in mapping.items():
            if new in target_atoms:
                legacy[old] = target_atoms[new]
        
        bridge_timeline = defaultdict(list)
        res_keys = list(legacy.keys())
        
        for idx in frames:
            coords = traj.xyz[idx] * 10
            water_coords = coords[waters]
            frame_bridges = []
            
            for i, r1 in enumerate(res_keys):
                for j, r2 in enumerate(res_keys[i+1:], i+1):
                    c1 = coords[legacy[r1]]
                    c2 = coords[legacy[r2]]
                    
                    d1 = cdist(water_coords, c1).min(axis=1)
                    d2 = cdist(water_coords, c2).min(axis=1)
                    
                    key = f"{r1}--{r2}"
                    has_bridge = np.any((d1 <= self.water_cutoff) & (d2 <= self.water_cutoff))
                    
                    bridge_timeline[key].append(1 if has_bridge else 0)
                    if has_bridge:
                        frame_bridges.append(key)
        
        # calculate occupancies
        bridge_data = {}
        for key, timeline in bridge_timeline.items():
            bridge_data[key] = {
                'occupancy': np.mean(timeline) if timeline else 0,
                'timeline': np.array(timeline)
            }
        
        return {
            'bridges': bridge_data,
            'boundaries': boundaries,
            'labels': labels
        }
    
    def run(self, traj_files, output_prefix="water_analysis"):
        """run complete analysis"""
        print("="*60)
        print("water-mediated interaction analysis")
        print("="*60)
        start = time.time()
        
        # load data
        traj, boundaries, labels = self.load_trajectories(traj_files)
        waters, target_atoms = self.find_atoms(traj)
        
        # sample frames
        frames = list(range(0, traj.n_frames, self.frame_step))
        print(f"\nsampling every {self.frame_step}th frame: {len(frames)} total")
        
        # run analyses
        self.results['distances'] = self.calc_distances(traj, target_atoms, frames)
        self.results['rdf'] = self.calc_rdf(traj, target_atoms, waters, frames)
        self.results['interactions'] = self.calc_interactions(traj, target_atoms, waters, frames)
        self.results['bridges'] = self.calc_bridges(traj, target_atoms, waters, frames)
        self.results['legacy_bridges'] = self.calc_legacy_bridges(traj, target_atoms, waters, frames, boundaries, labels)
        
        # save and plot
        self.save_results(output_prefix)
        self.create_plots(output_prefix)
        
        elapsed = time.time() - start
        print(f"\nanalysis completed in {elapsed:.1f} seconds")
        
        return self.results
    
    def save_results(self, prefix):
        """save results to files"""
        print("\nsaving results...")
        
        with open(f"{prefix}_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        with open(f"{prefix}_summary.txt", 'w') as f:
            f.write("water analysis summary\n")
            f.write("="*60 + "\n\n")
            
            f.write("distances:\n")
            for pair, data in self.results['distances'].items():
                mean = np.mean(data['distances'])
                std = np.std(data['distances'])
                f.write(f"  {pair}: {mean:.2f} +/- {std:.2f} A\n")
            
            f.write("\ninteraction types:\n")
            for pair, data in self.results['interactions'].items():
                f.write(f"  {pair}:\n")
                f.write(f"    direct: {data['direct']*100:.1f}%\n")
                f.write(f"    water-mediated: {data['water']*100:.1f}%\n")
                f.write(f"    no interaction: {data['none']*100:.1f}%\n")
            
            f.write("\nwater bridges:\n")
            for bridge, data in self.results['bridges'].items():
                f.write(f"  {bridge}: {data['occupancy']*100:.1f}%\n")
        
        print(f"  saved to {prefix}_results.pkl and {prefix}_summary.txt")
    
    def create_plots(self, prefix):
        """generate plots"""
        print("\ncreating plots...")
        
        # distance plots
        dist_data = self.results['distances']
        n = len(dist_data)
        
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.5*n))
        if n == 1:
            axes = [axes]
        
        for i, (pair, data) in enumerate(dist_data.items()):
            axes[i].plot(data['time'], data['distances'], linewidth=1, alpha=0.8, color='steelblue')
            axes[i].axhline(3.5, color='red', linestyle='--', alpha=0.7, label='direct (3.5 A)')
            axes[i].axhline(6.0, color='orange', linestyle='--', alpha=0.7, label='water-mediated')
            
            mean = np.mean(data['distances'])
            std = np.std(data['distances'])
            axes[i].text(0.02, 0.98, f'mean: {mean:.1f} +/- {std:.1f} A',
                        transform=axes[i].transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_ylabel(f'{pair}\ndistance (A)')
            axes[i].set_title(f'distance: {pair.replace("-", " - ")}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            if i == n-1:
                axes[i].set_xlabel('time (ns)')
        
        plt.tight_layout()
        plt.savefig(f"{prefix}_distances.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # rdf plots
        if self.results['rdf']:
            rdf_data = self.results['rdf']
            n = len(rdf_data)
            
            fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
            if n == 1:
                axes = [axes]
            
            for i, (pair, data) in enumerate(rdf_data.items()):
                axes[i].plot(data['r'], data['gr'], linewidth=2, color='blue')
                axes[i].axhline(1, color='black', linestyle='--', alpha=0.5, label='bulk')
                axes[i].axvline(3.5, color='red', linestyle='--', alpha=0.7, label='cutoff')
                
                # find peaks
                if len(data['gr']) > 0 and np.max(data['gr']) > 1.1:
                    peaks, _ = find_peaks(data['gr'], height=1.1, distance=3)
                    if len(peaks) > 0:
                        axes[i].plot(data['r'][peaks], data['gr'][peaks], 'ro', markersize=6)
                        for p in peaks[:2]:
                            axes[i].text(data['r'][p], data['gr'][p] + 0.1,
                                       f'{data["r"][p]:.1f} A', ha='center', fontsize=8)
                
                axes[i].set_xlabel('distance (A)')
                axes[i].set_ylabel('g(r)')
                axes[i].set_title(f'rdf: {pair.replace("-", " - ")}')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                axes[i].set_xlim(0, 8)
                
                axes[i].text(0.02, 0.98, f'n = {data["n_distances"]}',
                           transform=axes[i].transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{prefix}_rdf.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # bridge comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        bridges = {k: v for k, v in self.results['bridges'].items() if v['occupancy'] >= 0.1}
        if bridges:
            names = list(bridges.keys())
            occs = [bridges[n]['occupancy'] for n in names]
            
            bars = ax1.bar(range(len(names)), occs, alpha=0.7, color='steelblue')
            ax1.set_xlabel('water bridges')
            ax1.set_ylabel('occupancy')
            ax1.set_title('current analysis')
            ax1.set_xticks(range(len(names)))
            ax1.set_xticklabels([n.replace('--', ' - ') for n in names], rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            for bar, occ in zip(bars, occs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{occ:.2f}', ha='center', va='bottom', fontweight='bold')
        
        legacy = {k: v for k, v in self.results['legacy_bridges']['bridges'].items() if v['occupancy'] >= 0.1}
        if legacy:
            names = list(legacy.keys())
            occs = [legacy[n]['occupancy'] for n in names]
            
            bars = ax2.bar(range(len(names)), occs, alpha=0.7, color='coral')
            ax2.set_xlabel('water bridges')
            ax2.set_ylabel('occupancy')
            ax2.set_title('legacy analysis')
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels([n.replace('--', ' - ').replace('_', ' ') for n in names], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            for bar, occ in zip(bars, occs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{occ:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{prefix}_bridges.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # interaction types
        int_data = self.results['interactions']
        pairs = list(int_data.keys())
        direct = [int_data[p]['direct'] for p in pairs]
        water = [int_data[p]['water'] for p in pairs]
        none = [int_data[p]['none'] for p in pairs]
        
        x = np.arange(len(pairs))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, direct, width, label='direct', color='coral', alpha=0.8)
        plt.bar(x, water, width, label='water-mediated', color='skyblue', alpha=0.8)
        plt.bar(x + width, none, width, label='no interaction', color='lightgreen', alpha=0.8)
        
        plt.xlabel('residue pairs')
        plt.ylabel('fraction')
        plt.title('interaction types')
        plt.xticks(x, [p.replace('-', ' - ') for p in pairs], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # add labels
        for i in range(len(pairs)):
            if direct[i] > 0.05:
                plt.text(i - width, direct[i] + 0.01, f'{direct[i]:.1%}',
                        ha='center', va='bottom', fontsize=8)
            if water[i] > 0.05:
                plt.text(i, water[i] + 0.01, f'{water[i]:.1%}',
                        ha='center', va='bottom', fontsize=8)
            if none[i] > 0.05:
                plt.text(i + width, none[i] + 0.01, f'{none[i]:.1%}',
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{prefix}_interactions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  saved plots with prefix {prefix}")
    
    def print_summary(self):
        """print analysis summary"""
        print("\n" + "="*60)
        print("analysis summary")
        print("="*60)
        
        print("\ndistances:")
        for pair, data in self.results['distances'].items():
            mean = np.mean(data['distances'])
            std = np.std(data['distances'])
            print(f"  {pair}: {mean:.2f} +/- {std:.2f} A")
        
        print("\ninteraction types:")
        for pair, data in self.results['interactions'].items():
            print(f"  {pair}:")
            print(f"    direct: {data['direct']*100:.1f}%")
            print(f"    water-mediated: {data['water']*100:.1f}%")
            print(f"    no interaction: {data['none']*100:.1f}%")
        
        print("\nwater bridges:")
        bridges = [(k, v['occupancy']) for k, v in self.results['bridges'].items() if v['occupancy'] >= 0.1]
        bridges.sort(key=lambda x: x[1], reverse=True)
        for name, occ in bridges:
            print(f"  {name}: {occ*100:.1f}%")


if __name__ == "__main__":
    # trajectory files
    traj_files = {
        "D20": ("D20.dcd", "ROS_1_SC.psf"),
        "D21": ("D21.dcd", "ROS_1_SC.psf"),
        "D22": ("D22.dcd", "ROS_1_SC.psf")
    }
    
    try:
        analyzer = WaterAnalyzer()
        results = analyzer.run(traj_files, "water_analysis")
        analyzer.print_summary()
        
        print("\nfiles generated:")
        print("- water_analysis_results.pkl")
        print("- water_analysis_summary.txt")
        print("- water_analysis_distances.png")
        print("- water_analysis_rdf.png")
        print("- water_analysis_bridges.png")
        print("- water_analysis_interactions.png")
        
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
