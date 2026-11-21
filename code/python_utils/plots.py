from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
import numpy as np
from dataclasses import dataclass
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
plt.style.use(PROJECT_ROOT/"code/python_utils/stylesheet.mplstyle")
textwidth = 16/2.54
oogr = 1/1.61803398875

class lattice_type:
    LINEAR     = 0
    SQUARE     = 1
    TRIANGULAR = 2

class boundary_conditions:
    PERIODIC = 0
    OPEN     = 1

@dataclass
class lattice_params:
    ltype: int
    bcond: int
    N:     int
    ldim:  int
    Ns:    int

@dataclass
class simulation_params:
    n_cool:       int
    m:            int
    alpha:        float
    t_0:          float
    t_end:        float
    t0_trap:      float
    tau_trap:     float
    t0_res:       float
    tau_res:      float
    del_t:        float
    atol:      float
    rtol:      float
    max_stepsize: float

@dataclass
class boson_params:
    sigma:   int
    site_dof: int
    Np:      float
    j:       float
    u:       float
    v:       float
    g:       float
    p:       float
    rt:      float
    ut:      float
    beta:    float
    mu_eq:   float
    mu:      float

@dataclass
class fermion_params:
    sigma:   int
    site_dof: int
    filling: float
    j:       float
    u:       float
    v:       float
    g:       float
    q_scr:   float
    beta:    float
    mu_eq:   float
    mu:      float


def get_xy_from_n(n, ltype, N):
    if(ltype==0):
        return n
    elif(ltype==1):
        A = np.array([[1,0],[0,1]])
        i = n%N
        j = n//N
        return np.matmul(A,[i,j])

    elif(ltype==2):
        A = np.array([[1,0.5],[0,np.sqrt(3)/2]])
        i = n%N
        j = n//N
        return np.matmul(A,[i,j])

def MSD_from_occ(n_t, lp, pp, m):
    #note that this funciton is only valid for a fixed center of mass of the particle cloud, which is the case here
    N_t = np.shape(n_t)[0]
    msd_m = np.zeros(N_t)


    def compute_s0():
        tot_occ = 0
        res = np.zeros(lp.ldim)
        for n in range(lp.Ns):
            r = get_xy_from_n(n,lp.ltype, lp.N)

            loc_occ = np.sum(n_t[0,n,:])
            # loc_occ = n_t[0,n,m]
            # loc_occ = max(loc_occ, 1e-12) #assure that for an empty band, the center of mass is the center of the lattice and no division by 0 occurs

            res += loc_occ*r
            tot_occ += loc_occ
        res /= tot_occ
        return res

    s0 = compute_s0() #center of mass at the start
    N0 = np.sum(n_t[0,:,:]) #particle number at the start (whole system)

    def compute_Rsq(occ):
        tot_occ = 0
        res = 0
        for n in range(lp.Ns):
            r = get_xy_from_n(n,lp.ltype, lp.N)
            loc_occ = occ[n,m] #mas for m-th band
            # loc_occ = np.sum(occ[n,:]) #msd for all bands 


            res += loc_occ*np.linalg.norm(r-s0)**2
            tot_occ += loc_occ
            # tot_occ += np.sum(occ[n,:]) #normalize with N_tot(t)

        # tot_occ = np.sum(occ)
        if tot_occ < 1e-12:
            return 0
        else:
            res /= tot_occ #comment out for unnormalized msd
            # res /= N0
            return res

    R0sq = compute_Rsq(n_t[0,:,:])
    for j in range(1,N_t):
        # msd_m[j] = np.sqrt(np.abs(compute_Rsq(n_t[j,:,:])-R0sq))
        msd_m[j] = compute_Rsq(n_t[j,:,:])-R0sq #definition of experimentalists: no sqrt


    return msd_m
        
def write_scalar_trajectory(path, ft, times):
    if ft.size != times.size:
        return False
    
    ft = np.asarray(ft) #convert to np.arrays if they are not already
    times = np.asarray(times)

    try:
        with open(path, "wb") as out:
            time_steps = np.array(ft.size, dtype=np.int32)
            out.write(time_steps.tobytes())

            out.write(np.ascontiguousarray(times).tobytes())

            out.write(np.ascontiguousarray(ft).tobytes())
        return True
    except OSError:
        return False

def write_vector_trajectory(path, vt: np.ndarray, times: np.ndarray) -> bool:
 
    try:
        with open(path, 'wb') as f:
            time_steps = np.shape(vt)[0] 
            dim = np.shape(vt)[1] 

            header = np.array([dim, time_steps], dtype=np.int32)
            header.tofile(f)
            
            np.ascontiguousarray(times).tofile(f)
            np.ascontiguousarray(vt).tofile(f)
            
        return True
    except IOError:
        return False

def read_meta(path):
    with open(path, 'rb') as f:

        lp = np.fromfile(f, dtype=np.int32, count=5)
        lat_param = lattice_params(*lp)
        sp = np.fromfile(f, dtype=np.float64, count=13)
        sim_param = simulation_params(int(sp[0]), int(sp[1]), *sp[2:])
        sigma = int(np.fromfile(f, dtype=np.float64, count=1)[0])

        if sigma==1:
            bp = np.fromfile(f, dtype=np.float64, count=12)
            bos_param = boson_params(1, int(bp[0]), *bp[1:])
            return lat_param, sim_param, bos_param
        elif sigma==-1:
            fp = np.fromfile(f, dtype=np.float64, count=10)
            ferm_param = fermion_params(-1, int(fp[0]), *fp[1:])
            return lat_param, sim_param, ferm_param
        else:
            raise ValueError(f"Unknown sigma value: {sigma}")

def read_greens_trajectory(path):
    with open(path, 'rb') as f:

        dim, N_t = np.fromfile(f, dtype=np.int32, count=2)

        times = np.fromfile(f, dtype=np.float64, count=N_t)

        data = np.fromfile(f, dtype=np.complex128)
        
        # 3. Reshape and Correct Layout
        # Eigen (Column-Major) -> sequential read into (dim, dim) 
        # results in a Transposed matrix in Python's default C-style.
        # We reshape to (time, dim_cols, dim_rows) then swap axes.
        iGt = data.reshape((N_t, dim, dim)).transpose(0, 2, 1)

        return iGt, times

def read_vector_trajectory(path):
    with open(path, 'rb') as f:

        dim, N_t = np.fromfile(f, dtype=np.int32, count=2)

        times = np.fromfile(f, dtype=np.float64, count=N_t)

        data = np.fromfile(f, dtype=np.float64)
        

        vec = data.reshape((N_t, dim))

        return vec, times
    
def read_scalar_trajectory(path):
    with open(path, 'rb') as f:
        N_t = np.fromfile(f, dtype=np.int32, count=1)[0]

        times = np.fromfile(f, dtype=np.float64, count=N_t)

        ft = np.fromfile(f, dtype=np.float64)

        return ft, times

def convert_GtoOcc(
    subfolder = "",
    filename = "f_G.txt",
    meta_filename = "f_meta.txt"
):
    iGt, times = read_greens_trajectory(PROJECT_ROOT/"data"/subfolder/filename)
    lp, sp, pp = read_meta(PROJECT_ROOT/"data"/subfolder/meta_filename)
    n = pp.sigma*np.real(np.diagonal(iGt, axis1=1, axis2=2))
    n = np.abs(n)
    n_tot = (n.reshape(n.shape[0], -1, pp.site_dof).sum(axis=2))

    write_vector_trajectory(PROJECT_ROOT/"data"/subfolder/"f_occs.txt", n_tot, times)

def readin_occs(
        subfolder = "sweep_final",
        filename = "f_occs.txt",
        filename_meta = "f_meta.txt"
        ):
    n, times = read_vector_trajectory(PROJECT_ROOT/ "data"/subfolder/filename)
    lp, sp, pp = read_meta(PROJECT_ROOT/ "data"/subfolder/filename_meta)

    N_times, N_sites = n.shape

    x = np.zeros(N_sites)
    y = np.zeros(N_sites)
    for n_idx in range(N_sites):
        xy_n = get_xy_from_n(n_idx, lp.ltype, lp.N)
        x[n_idx] = xy_n[0]
        y[n_idx] = xy_n[1]

    return n, x, y, times


def plot_occ_trajectory(
    subfolder="",
    save_path=None
):
    
    indcs = [34, 37]

    occ_files = [("b_occs_"+str(i)+".txt") for i in indcs]
    meta_files = [("b_meta_"+str(i)+".txt") for i in indcs]
   
    N_times = 3
    N_systems = len(occ_files)

    n_s = np.empty(N_systems, dtype=object)
    times = np.empty(N_systems, dtype=object)
    lps = np.empty(N_systems, dtype=object)
    sps = np.empty(N_systems, dtype=object)
    pps = np.empty(N_systems, dtype=object)
  
    fills = np.zeros(N_systems)
    _, nus = read_scalar_trajectory(PROJECT_ROOT/"data"/subfolder/"mus_nus.txt")


    for i in range(N_systems):
        lps[i], sps[i], pps[i] = read_meta(PROJECT_ROOT/"data"/subfolder/meta_files[i])

        n, times[i] = read_vector_trajectory(PROJECT_ROOT/"data"/subfolder/occ_files[i])
        n = np.abs(n)
        n_s[i] = n.reshape(n.shape[0], -1, pps[i].site_dof)[:,:,1]

        fills[i] = nus[indcs[i]]


    t_min = times[0][0]
    t_max = 2.0
    t = np.linspace(t_min, t_max, N_times)   

    if(lps[0].ltype==0):
        fig, ax = plt.subplots(1, 
            N_systems, 
            figsize=(12, 4),
            layout="compressed")
         
        if N_systems == 1:
            ax = np.expand_dims(ax, axis=0)

        cmap = plt.get_cmap("inferno")
        norm = plt.Normalize(vmin=t_min, vmax=t_max)
        for system in range(N_systems):
            for timepoint in range(N_times):
                idx = np.argmin(abs(times[system] - t[timepoint]))
                color = cmap(norm(times[system][idx]))
                ax[system].plot(n_s[system][idx, :], color=color)
            ax[system].set_xlabel("site")

        ax[0].set_ylabel(r"$n_{i}$")

        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label=f"$t{{\\cdot}}J_2$")
        cbar.formatter = FormatStrFormatter('%.2f')
        cbar.update_ticks()


        if save_path:
            plt.savefig(save_path)
    else:

        fig, ax = plt.subplots(
            N_systems, 
            N_times,
            layout = "compressed"
        )
        if N_systems == 1:
            ax = np.expand_dims(ax, axis=0)

        cmap = plt.get_cmap("gnuplot")
        
        temp1 = []
        temp2 = []
        for timepoint in range(N_times):
            for system in range(N_systems):
                idx = np.argmin(abs(times[system] - t[timepoint]))
                temp1.append(np.min(n_s[system][idx, :]))
                temp2.append(np.max(n_s[system][idx, :]))
            
        v_min = np.min(temp1)
        v_max = np.max(temp2)

        norm = PowerNorm(gamma=1.0, vmin=v_min, vmax=v_max)

        
        # 2. Loop through a grid to populate the plots
        for system in range(N_systems):
            Ns = lps[system].Ns
            x = np.zeros(Ns)
            y = np.zeros(Ns)
            for n in range(Ns):
                xy_n = get_xy_from_n(n, lps[system].ltype, lps[system].N)
                x[n] = xy_n[0]
                y[n] = xy_n[1]

            marker_size = (80/np.max(x))**2
            box_ratio = (y.max() - y.min()) / (x.max() - x.min())

            for timepoint in range(N_times):
                current_ax = ax[system, timepoint]
                
                idx = np.argmin(abs(times[system] - t[timepoint]))#np.argmin(abs(times[system] - t[timepoint]))
                
                # NOTE: You would swap n_spsym out for different data sets depending on the 'system'
                current_ax.scatter(x, y, c=n_s[system][idx, :], cmap=cmap, norm=norm, s=marker_size)
                
                # Strip all ticks and enforce square aspect ratio
                current_ax.set_xticks([])
                current_ax.set_yticks([])
                current_ax.set_box_aspect(box_ratio)  
                
                # 3. Add Column Headers (Time) ONLY to the top system
                if system == 0:
                    current_ax.set_xlabel(f"$t{{\\cdot}} J_2$ = {times[system][idx]:.2f}", fontsize=12)
                    current_ax.xaxis.set_label_position('top')

                # 4. Add Row Labels (Parameters) ONLY to the first column
                if timepoint == 0:
                    current_ax.set_ylabel(f"$\\nu_e\\, = {fills[system]:.2f}$",fontsize=12, labelpad=10)
                    continue

        # 5. Create a single, shared colorbar on the right
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, 
            ax=ax.ravel().tolist(), 
            location="bottom",
        )
        cbar.set_label(r"$n_{i2}(t)$")
        cbar.formatter = FormatStrFormatter('%.1f')
        cbar.update_ticks()

        ticks = np.linspace(v_min, v_max, 6)
        digits = 3
        ticks[1:-1] = np.round(ticks[1:-1], digits)
        ticks[0] = v_min
        ticks[-1] = v_max
        cbar.set_ticks(ticks)

        if save_path:
            plt.savefig(save_path)

def plot_diam_trajectory(
    subfolder = "",
    save_path=None
):
    indcs = [26,32,36]


    diam_files = [f for f in ["b_diams_"+str(n)+".txt" for n in indcs]]
    meta_files = [f for f in ["b_meta_"+str(n)+".txt" for n in indcs]]


    N_systems = len(diam_files)

    diams = np.empty(N_systems, dtype=object)
    times = np.empty(N_systems, dtype=object)
    lps = np.empty(N_systems, dtype=object)
    sps = np.empty(N_systems, dtype=object)
    pps = np.empty(N_systems, dtype=object)
    fills = np.empty(N_systems, dtype=object)

    _, nus = read_scalar_trajectory(PROJECT_ROOT/"data"/subfolder/"mus_nus.txt")

    for i in range(N_systems):
        diams[i], times[i] = read_scalar_trajectory(PROJECT_ROOT/"data"/subfolder/diam_files[i])
        lps[i], sps[i], pps[i] = read_meta(PROJECT_ROOT/"data"/subfolder/meta_files[i])

        fills[i] = nus[indcs[i]]

    
    t_min = 0.0
    t_max = times[0][-1]

    fig, ax = plt.subplots()

    ymax = 0
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=fills[0], vmax=fills[-1])

    for system in range(N_systems):

        color =  cmap(norm(fills[system]))

        ax.plot(times[system], diams[system], label=f"$\\nu_e = {np.abs(fills[system]):.2f}$", color = color)#color=color)#/{indcs[system]}

        end_idx = np.argmin(np.abs(times[system]-t_max))
        ymax_sys = np.max(diams[system][:end_idx+1])
        ymax = max(ymax, ymax_sys)

    ax.set_xlabel(f"$t{{\\cdot}}J_2$")
    ax.set_ylabel(r"$\mathrm{MSD}(t)\,/\,a^2$")
    ax.set_xlim(t_min,t_max)
    ax.set_ylim(0, ymax*1.05)
    ax.legend()

    if save_path:
        plt.savefig(save_path)

def plot_diam_heatmap(
    quantity="MSD",
    subfolder = "",
    save_path=None
    ):
    _, its = read_scalar_trajectory(PROJECT_ROOT/"data"/subfolder/"converged_its.txt")
    indcs = its.astype(int)


    diam_files = [f for f in ["b_diams_"+str(n)+".txt" for n in indcs]]
    meta_files = ["b_meta_"+str(n)+".txt" for n in indcs]

    N_systems = len(diam_files)

    diams = np.empty(N_systems, dtype=object)
    times = np.empty(N_systems, dtype=object)
    lps = np.empty(N_systems, dtype=object)
    sps = np.empty(N_systems, dtype=object)
    pps = np.empty(N_systems, dtype=object)

    _, nus = read_scalar_trajectory(PROJECT_ROOT/"data"/subfolder/"mus_nus.txt")
    fills = np.empty(N_systems)


    for i in range(N_systems):
        diams[i], times[i] = read_scalar_trajectory(PROJECT_ROOT/"data"/subfolder/diam_files[i])
        lps[i], sps[i], pps[i] = read_meta(PROJECT_ROOT/"data"/subfolder/meta_files[i])
        fills[i] = nus[i]


    t_min = 0.0
    t_max = times[0][-1]
    x_min = fills[0]
    x_max = fills[-1]

    # --- interpolate every trajectory onto a common time grid ---
    n_time_pts = 400
    t_grid = np.linspace(t_min, t_max, n_time_pts)


    Z = np.empty((n_time_pts, N_systems))
    for system in range(N_systems):
        Z[:, system] = np.interp(
            t_grid,
            times[system],
            diams[system],
        )

    # --- sort systems by filling so the x-axis is monotonic ---
    order = np.argsort(fills)
    fills_sorted = fills[order]
    Z_sorted = Z[:, order]

    # Check for duplicate filling values, which would break np.gradient.
    if np.any(np.diff(fills_sorted) <= 0):
        raise ValueError(
            "Filling values must be unique and strictly increasing."
        )

    # Points lying inside the requested visible interval.
    interval_mask = (
        (fills_sorted >= x_min)
        & (fills_sorted <= x_max)
    )

    if not np.any(interval_mask):
        raise ValueError(
            f"No filling values found in [{x_min}, {x_max}]."
        )

    # Include one additional point on either side for pcolormesh.
    # With shading='nearest', the supplied x values are cell centers.
    interval_indices = np.flatnonzero(interval_mask)

    i_start = max(interval_indices[0] - 1, 0)
    i_stop = min(interval_indices[-1] + 1, len(fills_sorted) - 1)

    plot_slice = slice(i_start, i_stop + 1)
    fills_plot = fills_sorted[plot_slice]

    if quantity == "MSD":
        # Full data, followed by the slightly enlarged plotting interval.
        Z_full = Z_sorted
        Z_plot = Z_full[:, plot_slice]

        # Normalize only using data inside the exact requested interval.
        Z_norm = Z_full[:, interval_mask]

        colorbar_label = r"$\mathrm{MSD}\,/\,a^2$"

        cmap = "coolwarm"

        v_min = np.min(Z_norm)
        v_max = np.max(Z_norm)

        norm = PowerNorm(
            gamma=1.0,
            vmin=v_min,
            vmax=v_max,
        )


    elif quantity == "dMSD":
        # Calculate the derivative over the complete x-domain first.
        Z_full = np.gradient(
            Z_sorted,
            fills_sorted,
            axis=1,
            edge_order=2,
        )


        # Plot one additional point on each side so boundary cells
        # are not cut off by ax.set_xlim().
        Z_plot = Z_full[:, plot_slice]
        # Normalize only using the exact requested interval.
        Z_norm = Z_full[:, interval_mask]

        colorbar_label = (
            r"$\frac{\partial\,\mathrm{MSD}}{\partial\nu_e} \,/\,a^2$"
        )
        cmap = "coolwarm"

        abs_limit = np.nanpercentile(
            np.abs(Z_norm),
            98,
        )

        if not np.isfinite(abs_limit) or abs_limit == 0:
            abs_limit = 1.0

        norm = TwoSlopeNorm(
            vmin=-abs_limit,
            vcenter=0.0,
            vmax=abs_limit,
        )

    else:
        raise ValueError(
            f"Unknown quantity {quantity!r}. "
            "Use quantity='msd' or quantity='dx'."
        )

    fig, ax = plt.subplots()

    mesh = ax.pcolormesh(
        fills_plot, t_grid, Z_plot,
        cmap=cmap, norm=norm, rasterized=True,
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel(r"$\nu_e$")
    ax.set_ylabel(r"$t{\cdot}J_2$")
    ax.set_ylim(t_min, t_max)
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()
    ax.grid(False)

    if save_path:
        fig.savefig(save_path)

def add_callout(fig, ax, x_target, y_target, inset_pos, plot_func, inset_anchor=(0.5, 0)):
    """
    Mark (x_target, y_target) on `ax`, add an inset axes at `inset_pos`
    (axes-fraction coords relative to `ax`; can extend outside [0, 1] to
    sit above/beside the main plot), fill it via `plot_func(axins)`, and
    draw a connecting line from the marked point to the inset.
 
    Returns (axins, plot_func's return value) so you can e.g. grab the
    scatter mappable for a colorbar.
    """
    ax.plot(x_target, y_target, "ko", zorder=5, ms=5)
    axins = ax.inset_axes(inset_pos)
    artist = plot_func(axins)
 
    con = ConnectionPatch(
        xyA=(x_target, y_target), coordsA=ax.transData,
        xyB=inset_anchor, coordsB=axins.transAxes,
        color="0.4", lw=1, zorder=4,
    )
    fig.add_artist(con)
    return axins, artist
 
def plot_configuration(ax, idx, x, y, n, cmap, norm, marker_size):
    """Your existing scatter code, unchanged, just targeting a given `ax`."""
    scat = ax.scatter(x, y, c=n[idx], cmap=cmap, norm=norm, s=marker_size)
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    return scat
 
def phase_analysis(
    subfolder="",
    filename="f_occs.txt",
    filename_meta="f_meta.txt",
    filename_musnus="mus_nus.txt",
    save_path=None,
    x_range=(4.5,5.7),
):
    mus, nus = read_scalar_trajectory(PROJECT_ROOT / "data" / subfolder / filename_musnus)
    
    n, x, y, _ = readin_occs(subfolder=subfolder, filename=filename, filename_meta=filename_meta)
    _, _ , fp = read_meta(PROJECT_ROOT/"data"/subfolder/"f_meta.txt")
    mus /= fp.j
    x_range /= fp.j
 
    if x_range is not None:
        mask = (mus >= x_range[0]) & (mus <= x_range[1])
        mus_plot, nus_plot = mus[mask], nus[mask]
    else:
        mus_plot, nus_plot = mus, nus
 
    fig, ax = plt.subplots()
    ax.plot(mus_plot, nus_plot, color="blue")
    ax.set_xlabel(r"$\mu \,/\,J_1$")
    ax.set_ylabel(r"$\nu_e$")
    ax.margins(x=0)
 
    # shared colormap/norm so both insets (and the colorbar) are comparable
    cmap = plt.get_cmap("gnuplot")
    v_min, v_max = np.min(n), np.max(n)
    norm = PowerNorm(gamma=0.7, vmin=v_min, vmax=v_max)
    marker_size = (60/np.max(x))**2


    idcs = [34, 37]

    inset_size = 0.4

    rel_pos = [(mus[i]-mus_plot[0])/(mus_plot[-1]-mus_plot[0])-inset_size/2 for i in idcs]
    inset_positions = []
    inset_anchors = []
    for i in range(len(idcs)):
        if (i%2)==0:
            inset_positions.append([rel_pos[i], 1-inset_size, inset_size, inset_size])
            inset_anchors.append((0.5,0))
        else:
            inset_positions.append([rel_pos[i], 0, inset_size, inset_size])
            inset_anchors.append((0.5,1))


    scat = None
    for idx, pos, anchor in zip(idcs, inset_positions, inset_anchors):
        _, scat = add_callout(
            fig, ax, mus[idx], nus[idx], pos,
            lambda a, idx=idx: plot_configuration(a, idx, x, y, n, cmap, norm, marker_size),
            inset_anchor=anchor,
        )
 
    # one shared colorbar, since both insets use the same norm/cmap
    cbar = fig.colorbar(scat, ax=ax, aspect=15, pad=0.04)
    cbar.set_label(r"$n_{i}$")
    ticks = np.linspace(v_min, v_max, 6)
    ticks[1:-1] = np.round(ticks[1:-1], 1)
    cbar.set_ticks(ticks)
    cbar.formatter = FormatStrFormatter('%.2f')
    cbar.update_ticks()

    if save_path:
        plt.savefig(save_path)


# convert_GtoOcc()

plot_occ_trajectory()
plot_diam_trajectory()
plot_diam_heatmap()
phase_analysis()



plt.show()