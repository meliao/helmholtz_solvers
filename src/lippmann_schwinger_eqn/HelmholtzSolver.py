import logging
import numpy as np

import torch
import cola

from typing import Tuple, Dict

from src.lippmann_schwinger_eqn.solver_utils import (
    greensfunction2,
    greensfunction3,
    getGscat2circ,
    find_diag_correction,
    get_extended_grid,
)
from src.lippmann_schwinger_eqn.bicgstab_batch import bicgstab_batch

logging.getLogger("cola-ml").setLevel(logging.WARNING)
logging.getLogger("cola").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)
logging.getLogger("plum").setLevel(logging.WARNING)
logging.getLogger("plum-dispatch").setLevel(logging.WARNING)
logging.getLogger("plum-dispatch").setLevel(logging.WARNING)


class HelmholtzSolverBase:
    def __init__(
        self,
        domain_points: np.ndarray,
        frequency: float,
        exterior_greens_function: np.ndarray,
        N: int,
        source_dirs: np.ndarray,
        x_vals: np.ndarray,
        use_bicgstab: bool = False,
        max_iter_bicgstab: int = 10,
        diag_correction: float = None,
    ) -> None:
        self.domain_points = torch.from_numpy(domain_points).to(torch.float)
        self.frequency = frequency
        self.frequency_torch = torch.Tensor([frequency]).to(torch.float)

        self.N = N
        self.source_dirs = source_dirs
        self.x_vals = x_vals
        self.domain_points_arr = domain_points.reshape((N, N, 2))

        self.h = self.domain_points_arr[0, 1, 0] - self.domain_points_arr[0, 0, 0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exterior_greens_function = (
            torch.from_numpy(exterior_greens_function).to(torch.cfloat).to(self.device)
        )

        self.use_bicgstab = use_bicgstab
        self.max_iter_bicgstab = max_iter_bicgstab
        self.diag_correction = diag_correction

    def _get_uin(self, source_directions: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            source_directions (torch.Tensor): Has shape (n_directions,)

        Returns:
            torch.Tensor: Has shape (n_directions, self.N**2)
        """
        inc = torch.stack(
            [torch.cos(source_directions), torch.sin(source_directions)]
        ).to(torch.float)
        # print("_get_uin: inc shape: ", inc.shape)
        inner_prods = self.domain_points.to(self.device) @ inc
        # print("_get_uin: inner_prods shape: ", inner_prods.shape)

        uin = (
            torch.exp(1j * self.frequency * inner_prods).to(torch.cfloat).permute(1, 0)
        )
        return uin

    def _get_uin_sigma(
        self,
        source_directions: torch.Tensor,
        scattering_obj: torch.Tensor,
        return_infodict: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            source_direction (torch.Tensor): Has shape (n_directions,)
            scattering_obj (torch.Tensor): shape (N, N) OR (batch, N, N). If the
                latter, the output will also be batched.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First output is uin which has shape
            (n_directions, N**2). Second output is sigma, which has shape (n_directions, N**2) OR
            (batch, n_directions, N**2)
        """
        # print("_get_uin_sigma: scattering_obj shape", scattering_obj.shape)

        uin = self._get_uin(source_directions)

        # print("_get_uin_sigma: uin shape: ", uin.shape)

        if torch.all(scattering_obj == torch.zeros_like(scattering_obj)):
            logging.warning("All-zero scattering object encountered")
            sigma = torch.zeros(
                (source_directions.shape[0], self.N**2),
                dtype=torch.cfloat,
                device=scattering_obj.device,
            )
            out_info = {"niter": 0}

        else:
            if self.use_bicgstab:
                batched = True
                if scattering_obj.ndim == 2:
                    scattering_obj = scattering_obj.unsqueeze(0)
                    batched = False
                sigma, out_info = self._bicgstab_Helmholtz_inv(scattering_obj, uin)
                if not batched:
                    sigma = sigma.squeeze(0)
            else:
                if scattering_obj.ndim == 3:
                    raise ValueError(
                        f"Scattering object has shape {scattering_obj.shape} which is incompatible with GMRES method."
                    )
                sigma, out_info = self._gmres_Helmholtz_inv(scattering_obj, uin)

        if return_infodict:
            return uin, sigma, out_info
        else:
            return uin, sigma

    def _G_apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the green's function operator. This is used as a subroutine in
        GMRES calls.

        Args:
            x (np.ndarray): Has shape (N ** 2,)

        Raises:
            NotImplementedError: This must be implemented by child classes

        Returns:
            np.ndarray: Has shape (N ** 2,)
        """
        raise NotImplementedError()

    def _gmres_Helmholtz_inv(
        self,
        scattering_obj: torch.Tensor,
        uin: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """_summary_

        Args:
            scattering_obj (torch.Tensor): Has shape (N, N)
            uin (torch.Tensor): Has shape (n_directions, self.N**2)
            to be on this device and will compute on this device.

        Returns:
            Tuple[torch.Tensor, Dict]: The tensor is the result; it
                has shape (n_directions, self.N**2)
                The dict is information from the solver.
        """

        n = scattering_obj.shape[0]
        q = scattering_obj.flatten().unsqueeze(-1)

        def _matvec(x: torch.Tensor) -> torch.Tensor:
            # print("_matvec: input shape: ", x.shape)
            gout = self._G_apply(x)
            term2 = (self.frequency**2) * q * gout
            # print("_matvec: term2 shape and device: ", term2.shape, term2.device)
            y = x + term2.to(torch.cfloat)
            # print("_matvec: output device: ", y.device)
            # print("_matvec: output shape: ", y.shape)
            return y

        A = cola.ops.LinearOperator(
            torch.complex64,
            (self.N**2, self.N**2),
            matmat=_matvec,
        )

        # X = A.to(self.device)
        A.device = self.device
        # print("_gmres_Helmholtz_inv: new A operator on device: ", A.device)
        b = -(self.frequency**2) * q * uin.permute(1, 0)
        b = b.to(torch.cfloat)

        sigma, out_info = cola.algorithms.gmres(A, b)

        out = sigma.permute(1, 0)

        logging.warning(
            f"_gmres_Helmholtz_inv: GMRES exited after {out_info['iterations']} iterations"
            f" with a final error of {out_info['errors'][-1]:.4e}"  # Is this the right error?
        )
        # print("_gmres_Helmholtz_inv: output shape: ", out.shape)
        return out, out_info

    def _bicgstab_Helmholtz_inv(
        self, scattering_obj: torch.Tensor, uin: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """This inverts the Helmholtz equation for a batch of uin directions
        using BICGSTAB.

        Args:
            scattering_obj (torch.Tensor): Has shape (self.N, self.N)
            uin (torch.Tensor): Has shape (n_directions, self.N**2)

        Returns:
            Tuple[torch.Tensor, Dict]: The tensor is the result; it
                has shape (n_directions, self.N**2)
                The dict is information from the solver.
        """
        K, n, _ = scattering_obj.shape
        # print("_bicgstab_Helmholtz_inv: scattering_obj shape", scattering_obj.shape)
        q = scattering_obj.reshape(K, n**2, 1)  # .to(torch.cdouble)
        # print("_bicgstab_Helmholtz_inv: q shape", q.shape)

        def _matvec(x: torch.Tensor) -> torch.Tensor:
            """Expects input shape (K, self.N**2, n_directions)
            and output should have shape (K, self.N**2, n_directions)
            """
            # print("_matvec: input shape: ", x.shape)
            gout = self._G_apply(x)
            term2 = (self.frequency**2) * q * gout
            # print("_matvec: term2 shape and device: ", term2.shape, term2.device)
            y = x + term2.to(torch.cfloat)
            # y = x + term2
            # print("_matvec: output device: ", y.device)
            # print("_matvec: output shape: ", y.shape)
            return y

        # diag_vec = torch.ones_like(q) + q * (self.frequency**2) * self.diag_correction

        # def _diag_precond(x: torch.Tensor) -> torch.Tensor:
        #     """Expects input shape (1, self.N**2, n_directions)
        #     and output should have shape (1, self.N**2, n_directions)
        #     """
        #     return x / diag_vec

        # print("_gmres_Helmholtz_inv: new A operator on device: ", A.device)
        b = -(self.frequency**2) * q * uin.permute(1, 0)
        b = b.to(torch.cfloat)
        # print("_bicgstab_Helmholtz_inv: b shape", b.shape)

        # Initialize by doing b * (1 + 1 / q), but 1 / q might have nans.
        # We will zero out the nans.
        # x_init = b * (1 + 1 / q)
        # x_init[torch.isnan(x_init)] = 0
        # x_init = x_init.to(torch.cfloat)
        # x_init = b
        x_init = None

        sigma, out_info = bicgstab_batch(
            _matvec,
            b,
            maxiter=self.max_iter_bicgstab,
            # K_1_inv_bmm=_diag_precond,
            rtol=1e-04,
            atol=1e-04,
            X0=x_init,
        )
        # print(
        #     "_bicgstab_Helmholtz_inv: sigma shape and dtype: ", sigma.shape, sigma.dtype
        # )

        out = sigma.permute(0, 2, 1)
        # print("_bicgstab_Helmholtz_inv: out shape", out.shape)

        logging.debug(
            f"_bicgstab_Helmholtz_inv: BICGSTAB exited after {out_info['niter']} iterations"
        )
        # print("_gmres_Helmholtz_inv: output shape: ", out.shape)
        return out, out_info

    def Helmholtz_solve_exterior(
        self,
        source_directions: np.ndarray,
        scattering_obj: np.ndarray,
        numpy_cpu_output: bool = True,
        return_infodict: bool = False,
    ) -> np.ndarray:
        """Solve the Helmholtz equation on the exterior ring for a given source
        directions and a given scattering object. This function returns the scattered
        wave field, NOT the full wave field.

        Args:
            source_directions (np.ndarray): Angles in radians. Has shape (N_dirs, )
            scattering_obj (np.ndarray): Has shape (N, N) OR (batch, N, N). If the
                latter, the output will also be batched.

        Returns:
            np.ndarray: Has shape (N_dirs, N) OR (batch, N_dirs, N)
        """
        if torch.is_tensor(source_directions):
            directions_torch = source_directions.to(self.device)
        else:
            directions_torch = torch.from_numpy(source_directions).to(self.device)
        if torch.is_tensor(scattering_obj):
            scattering_obj_torch = scattering_obj.to(self.device)
        else:
            scattering_obj_torch = torch.from_numpy(scattering_obj).to(self.device)

        batched = False
        if scattering_obj_torch.ndim == 3:
            batched = True

        if torch.any(torch.isnan(scattering_obj_torch)):
            out_nan = torch.full(
                (source_directions.shape[0], self.N), torch.nan, dtype=torch.complex64
            )
            logging.warning("HelmholtzSolver: input scattering obj has NaNs")
            return out_nan

        if return_infodict:
            _, sigma, out_info = self._get_uin_sigma(
                directions_torch, scattering_obj_torch, return_infodict=True
            )
        else:
            _, sigma = self._get_uin_sigma(
                directions_torch, scattering_obj_torch, return_infodict=False
            )
        # print("Helmholtz_solve_exterior: sigma shape: ", sigma.shape)
        # print(
        #     "Helmholtz_solve_exterior: exterior_greens_function shape: ",
        #     self.exterior_greens_function.shape,
        # )

        if batched:
            FP = self.exterior_greens_function.to(self.device).to(
                sigma.dtype
            ) @ sigma.permute(0, 2, 1)
            out = FP.permute(0, 2, 1)
        else:
            FP = self.exterior_greens_function.to(self.device).to(
                sigma.dtype
            ) @ sigma.permute(1, 0)
            # FP = np.reshape(FP, (1, -1))
            out = FP.permute(1, 0)

        if numpy_cpu_output:
            out = out.cpu().numpy()

        if return_infodict:
            return out, out_info
        else:
            return out

    def Helmholtz_solve_interior(
        self,
        source_directions: np.ndarray,
        scattering_obj: np.ndarray,
        return_infodict: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_

        Args:
            source_direction (float): Has shape (N_dirs,)
            scattering_obj (np.ndarray): Has shape (N, N) OR (batch, N, N). If the
                latter, the output will also be batched.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (u_tot, u_in, u_scat).
            Each has shape (N_dirs, N, N) OR (batch, N_dirs, N, N)
        """
        n_directions = source_directions.shape[0]
        directions_torch = torch.from_numpy(source_directions).to(self.device)
        scattering_obj_torch = torch.from_numpy(scattering_obj).to(self.device)
        if scattering_obj_torch.ndim == 2:
            batched = False
        else:
            batched = True
            n_batch = scattering_obj_torch.shape[0]

        if return_infodict:
            uin, sigma, out_info = self._get_uin_sigma(
                directions_torch, scattering_obj_torch, return_infodict=True
            )
        else:
            uin, sigma = self._get_uin_sigma(
                directions_torch, scattering_obj_torch, return_infodict=False
            )
        # print("Helmholtz_solve_interior: uin shape: ", uin.shape)
        # print("Helmholtz_solve_interior: sigma shape: ", sigma.shape)
        if batched:
            u_s = self._G_apply(sigma.permute(0, 2, 1)).permute(0, 2, 1)
            out_shape_1 = (n_batch, n_directions, self.N, self.N)
            out_shape_2 = (n_directions, self.N, self.N)

        else:
            u_s = self._G_apply(sigma.permute(1, 0)).permute(1, 0)
            out_shape_2 = (n_directions, self.N, self.N)
            out_shape_1 = (n_directions, self.N, self.N)
        # print("Helmholtz_solve_interior: u_s shape: ", u_s.shape)
        # print("Helmholtz_solve_interior: u_in shape: ", uin.shape)
        u_tot = u_s + uin
        # print("Helmholtz_solve_interior: u_tot shape: ", u_tot.shape)
        if return_infodict:
            return (
                u_tot.reshape(out_shape_1).cpu().numpy(),
                uin.reshape(out_shape_2).cpu().numpy(),
                u_s.reshape(out_shape_1).cpu().numpy(),
                out_info,
            )
        else:
            return (
                u_tot.reshape(out_shape_1).cpu().numpy(),
                uin.reshape(out_shape_2).cpu().numpy(),
                u_s.reshape(out_shape_1).cpu().numpy(),
            )

    def Helmholtz_solve_full(
        self,
        source_directions: np.ndarray,
        scattering_obj: np.ndarray,
        numpy_cpu_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the Helmholtz equation and returns:
        1. The scattered wave field on the exterior.
        2. The total wave field on the interior

        Args:
            source_directions (np.ndarray): has shape (N_dirs,)
            scattering_obj (np.ndarray): Has shape (N, N)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Return 1 has shape (N_dirs, N)
            Return 2 has shape (N_dirs, N, N).
        """
        if torch.is_tensor(source_directions):
            directions_torch = source_directions.to(self.device)
        else:
            directions_torch = torch.from_numpy(source_directions).to(self.device)
        if torch.is_tensor(scattering_obj):
            scattering_obj_torch = scattering_obj.to(self.device)
        else:
            scattering_obj_torch = torch.from_numpy(scattering_obj).to(self.device)

        if torch.any(torch.isnan(scattering_obj_torch)):
            out_nan_1 = torch.full(
                (source_directions.shape[0], self.N), torch.nan, dtype=torch.complex64
            )
            out_nan_2 = torch.full(
                (source_directions.shape[0], self.N, self.N),
                torch.nan,
                dtype=torch.complex64,
            )
            logging.warning("HelmholtzSolver: input scattering obj has NaNs")
            if numpy_cpu_output:
                return out_nan_1.cpu().numpy(), out_nan_2.cpu().numpy()
            else:
                return out_nan_1, out_nan_2

        # Set up and solve the linear system to get sigma
        uin, sigma = self._get_uin_sigma(
            directions_torch, scattering_obj_torch, return_infodict=False
        )

        # Convolve with the interior G's function to get the scattered field on
        # the interior
        u_s = self._G_apply(sigma.permute(1, 0)).permute(1, 0)
        u_tot = u_s + uin
        u_tot = u_tot.reshape((source_directions.shape[0], self.N, self.N))

        # Convolve with the exterior G's function to get the scattered field on
        # the exterior
        u_scat_ext = self.exterior_greens_function @ sigma.permute(1, 0)
        u_scat_ext = u_scat_ext.permute(1, 0)

        if numpy_cpu_output:
            return u_scat_ext.cpu().numpy(), u_tot.cpu().numpy()
        else:
            return u_scat_ext, u_tot


class HelmholtzSolverAccelerated(HelmholtzSolverBase):
    def __init__(
        self,
        domain_points: np.ndarray,
        extended_domain_points: np.ndarray,
        G_fft: np.ndarray,
        frequency: float,
        exterior_greens_function: np.ndarray,
        N: int,
        source_dirs: np.ndarray,
        x_vals: np.ndarray,
        use_bicgstab: bool = False,
        max_iter_bicgstab: int = 10,
        diag_correction: float = None,
    ) -> None:
        super().__init__(
            domain_points,
            frequency,
            exterior_greens_function,
            N,
            source_dirs,
            x_vals,
            use_bicgstab=use_bicgstab,
            max_iter_bicgstab=max_iter_bicgstab,
            diag_correction=diag_correction,
        )
        self.extended_domain_points = extended_domain_points
        self.G_fft = torch.from_numpy(G_fft).to(self.device)

    def _G_apply(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (self.N**2, n_dirs) and output has shape (self.N**2, n_dirs)"""
        # print("_fast_G_apply: input shape: ", x.shape)
        # print("_fast_G_apply: G_fft shape: ", self.G_fft.shape)
        x_shape = x.shape
        batched = False
        if x.ndim == 3:
            # Put the first dimension last so the reshape folds the directions
            # and batches together.
            x = x.permute(1, 2, 0)
            x_shape = x.shape
            batched = True
        x_square = x.reshape(self.N, self.N, -1)
        # x_square = x.reshape(self.N, self.N, x_shape[-1])
        x_pad = self._zero_pad(x_square, self.G_fft.shape[0])
        # print("_fast_G_apply: x_pad shape: ", x_pad.shape)

        x_fft = torch.fft.fft2(x_pad, dim=(0, 1))

        prod = torch.einsum("ab,abc->abc", self.G_fft, x_fft)
        # print("_fast_G_apply: prod shape", prod.shape)
        out_fft = torch.fft.ifft2(prod, dim=(0, 1))

        out = out_fft[: self.N, : self.N]

        o = out.reshape(x_shape)
        if batched:
            o = o.permute(2, 0, 1)

        # logging.debug("_fast_G_apply: output shape: %s", o.shape)
        # print("_fast_G_apply: output shape: ", o.shape)

        return o

    def _zero_pad(self, v: torch.Tensor, n: int) -> torch.Tensor:
        """v has shape (n_small, n_small, n_dirs) and output has shape (n, n, n_dirs)"""
        o = torch.zeros((n, n, v.shape[2]), dtype=v.dtype, device=self.device)
        o[: v.shape[0], : v.shape[1]] = v

        return o


def setup_accelerated_solver(
    n_pixels: int,
    spatial_domain_max: float,
    wavenumber: float,
    receiver_radius: float,
    diag_correction: bool = True,
    use_bicgstab: bool = False,
    max_iter_bicgstab: int = 10,
) -> HelmholtzSolverAccelerated:
    """Precomputes objects that are reused across different PDE solves.

    Args:
        n_pixels (int): The number of spatial points along each axis of the
                    scattering domain. Also, the number of source/receiver
                    directions.
        wavenumber (float): The wavenumber being used in the problem. This is
                    the number of waves across the spatial domain.

    Returns:
        Dict: _description_
    """

    frequency = 2 * np.pi * wavenumber
    source_receiver_directions = np.linspace(0, 2 * np.pi, n_pixels + 1)[:n_pixels]

    x = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    y = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y)
    domain_points_lst = np.array([X.flatten(), Y.flatten()]).T

    extended_domain_points_grid = get_extended_grid(n_pixels, h)

    receiver_points = (
        receiver_radius
        * np.array(
            [np.cos(source_receiver_directions), np.sin(source_receiver_directions)]
        ).T
    )

    # logging.debug(
    #     "precompute_objects: reciever_points shape: %s", reciever_points.shape
    # )

    if diag_correction:
        diag_correction_val = find_diag_correction(h, frequency)
    else:
        diag_correction_val = None

    G_int = greensfunction3(
        extended_domain_points_grid,
        frequency,
        diag_correction=diag_correction_val,
        dx=h,
    )
    G_int_fft = np.fft.fft2(G_int)
    # interior_greens_function = greensfunction2(domain_points_lst, frequency)

    exterior_greens_function = getGscat2circ(
        domain_points_lst, receiver_points, frequency, dx=h
    )
    # exterior_greens_function = None

    out = HelmholtzSolverAccelerated(
        domain_points_lst,
        extended_domain_points_grid,
        G_int_fft,
        frequency,
        exterior_greens_function,
        n_pixels,
        source_receiver_directions,
        x,
        use_bicgstab=use_bicgstab,
        max_iter_bicgstab=max_iter_bicgstab,
        diag_correction=diag_correction_val,
    )
    return out


class HelmholtzSolverDense(HelmholtzSolverBase):
    def __init__(
        self,
        domain_points: np.ndarray,
        interior_greens_function: np.ndarray,
        frequency: float,
        exterior_greens_function: np.ndarray,
        N: int,
        source_dirs: np.ndarray,
        x_vals: np.ndarray,
        use_bicgstab: bool = False,
        max_iter_bicgstab: int = 10,
        diag_correction: float = None,
    ) -> None:
        super().__init__(
            domain_points,
            frequency,
            exterior_greens_function,
            N,
            source_dirs,
            x_vals,
            use_bicgstab=use_bicgstab,
            max_iter_bicgstab=max_iter_bicgstab,
            diag_correction=diag_correction,
        )
        self.interior_greens_function = interior_greens_function

    def _G_apply(self, x: np.ndarray) -> np.ndarray:
        return self.interior_greens_function @ x


def setup_dense_solver(
    n_pixels: int,
    spatial_domain_max: float,
    wavenumber: float,
    receiver_radius: float,
    diag_correction: bool = True,
    use_bicgstab: bool = False,
    max_iter_bicgstab: int = 10,
) -> HelmholtzSolverDense:
    """Precomputes objects that are reused across different PDE solves.

    Args:
        n_pixels (int): The number of spatial points along each axis of the
                    scattering domain. Also, the number of source/receiver
                    directions.
        wavenumber (float): The wavenumber being used in the problem. This is
                    the number of waves across the spatial domain.

    Returns:
        Dict: _description_
    """

    frequency = 2 * np.pi * wavenumber
    source_receiver_directions = np.linspace(0, 2 * np.pi, n_pixels + 1)[:n_pixels]

    x = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    y = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    X, Y = np.meshgrid(x, y)
    domain_points_lst = np.array([X.flatten(), Y.flatten()]).T

    # xlrg = np.linspace(
    #     -3 * spatial_domain_max, 3 * spatial_domain_max, 3 * n_pixels, endpoint=False
    # )

    # zero_idx = np.argwhere(xlrg == 0)[0, 0]
    # xlrg_rolled = np.roll(xlrg, -zero_idx)
    # X_L, Y_L = np.meshgrid(xlrg_rolled, xlrg_rolled)
    # extended_domain_points_grid = np.stack((X_L, Y_L), axis=-1)

    receiver_points = (
        receiver_radius
        * np.array(
            [np.cos(source_receiver_directions), np.sin(source_receiver_directions)]
        ).T
    )

    # logging.debug(
    #     "precompute_objects: reciever_points shape: %s", reciever_points.shape
    # )

    h = x[1] - x[0]
    if diag_correction:
        diag_correction_val = find_diag_correction(h, frequency)
    else:
        diag_correction_val = None

    interior_greens_function = greensfunction2(
        domain_points_lst, frequency, diag_correction=diag_correction_val, dx=h
    )
    # interior_greens_function = greensfunction2(domain_points_lst, frequency)

    exterior_greens_function = getGscat2circ(
        domain_points_lst, receiver_points, frequency, dx=h
    )

    out = HelmholtzSolverDense(
        domain_points_lst,
        interior_greens_function,
        frequency,
        exterior_greens_function,
        n_pixels,
        source_receiver_directions,
        x,
        use_bicgstab=use_bicgstab,
        max_iter_bicgstab=max_iter_bicgstab,
        diag_correction=diag_correction_val,
    )
    return out
