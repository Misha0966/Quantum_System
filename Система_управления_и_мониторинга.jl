# Функция управления системой
function control_system!(rho::Matrix{ComplexF64}, Hs::Vector{Matrix{ComplexF64}}, c_ops::Vector{Matrix{ComplexF64}}, gamma::Float64, dt::Float64, steps::Int, accumulator_energy::Float64, max_capacity::Float64)
    for i in 1:steps
        println("Шаг $i: Эволюция системы")
        
        # Мониторинг текущего состояния
        monitor_system!(rho, accumulator_energy, [])

        # Эволюция плотностной матрицы
        H = Hs[Int(mod(i-1, length(Hs))) + 1]  # Выбор Гамильтониана
        rho = evolve_density_operator(rho, H, dt, c_ops, gamma)  # Эволюция
        
        # Зарядка накопителя
        energy_generated = time_dependent_energy(start_energy, final_energy, i, steps)
        charge_accumulator!(energy_generated)

        # Проверка и оптимизация энергии в накопителе
        if accumulator_energy > max_capacity
            println("Энергия превышает ёмкость накопителя. Регулировка работы реактора.")
        end
    end
end