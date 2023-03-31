<script lang="ts">
    import { onMount } from "svelte";
    import { io } from "socket.io-client"
    import GraphReport from "./GraphReport.svelte";
    const values: Array<{
        name: string,
        color: string,
        current: number,
        displayType: "string" | "graph"
    }> = [
        {
            name: "CPU",
            color: "#FF0000",
            current: 50,
            displayType: "graph"
        },
        {
            name: "GPU",
            color: "#00FF00",
            current: 50,
            displayType: "graph"
        },
        {
            name: "RAM",
            color: "#0000FF",
            current: 50,
            displayType: "graph"
        }
    ]
    const socket = io("ws://192.168.0.68:6002")
    // socket.on("")
    socket.on("device", (arg) => {
        values[0].current = (arg.cpu[0] / arg.cpu[1]) * 100
        values[1].current = (arg.gpu[0] / arg.gpu[1]) * 100
        values[2].current = (arg.ram[0] / arg.ram[1]) * 100
    })
</script>

<main>
	<div class="wrapper">
        {#each values as value}
            <GraphReport name={value.name} current={value.current} color={value.color}/>
        {/each}
	</div>
</main>

<style>
	main {
        width: 100%;
        height: 100%;
        background-color: antiquewhite;
        position: relative;
	}

    .wrapper {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: .3em;
    }
</style>