import {
    Beaker,
    Cpu,
    Database,
    HardDrive,
    Globe,
    Zap,
    ShieldCheck,
    Code2,
    Activity
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"

interface SidebarProps {
    stats: any
    status: any
}

export function Sidebar({ stats, status }: SidebarProps) {
    const getAgentStatus = (taskTag: string) => {
        return status?.current_task.includes(taskTag) ? "default" : "secondary"
    }

    return (
        <div className="w-64 border-r bg-card flex flex-col h-full shrink-0">
            <div className="p-6">
                <div className="flex items-center gap-2 mb-2">
                    <Beaker className="text-primary h-6 w-6" />
                    <h1 className="text-xl font-bold tracking-tight">Edison Hub</h1>
                </div>
                <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider">Scientific Command</p>
            </div>

            <ScrollArea className="flex-1 px-4">
                <div className="space-y-6 pb-6">
                    {/* Neural Grid */}
                    <div>
                        <div className="px-2 py-2 flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest">
                            <Activity className="h-3 w-3" /> Neural Grid
                        </div>
                        <div className="space-y-3 mt-2">
                            <div className="p-3 bg-muted/50 rounded-lg space-y-2 border">
                                <div className="flex justify-between text-[10px] font-bold uppercase text-muted-foreground">
                                    <span className="flex items-center gap-1.5"><Cpu className="h-3 w-3" /> Engine</span>
                                    <span>{stats?.cpu_percent || 0}%</span>
                                </div>
                                <div className="h-1 bg-background rounded-full overflow-hidden">
                                    <div className="h-full bg-primary" style={{ width: `${stats?.cpu_percent || 0}%` }} />
                                </div>
                            </div>

                            <div className="p-3 bg-muted/50 rounded-lg space-y-2 border">
                                <div className="flex justify-between text-[10px] font-bold uppercase text-muted-foreground">
                                    <span className="flex items-center gap-1.5"><Database className="h-3 w-3" /> Memory</span>
                                    <span>{stats?.memory_used_gb || 0}GB</span>
                                </div>
                                <div className="h-1 bg-background rounded-full overflow-hidden">
                                    <div className="h-full bg-secondary" style={{ width: `${stats?.memory_percent || 0}%` }} />
                                </div>
                            </div>
                        </div>
                    </div>

                    <Separator />

                    {/* Cognitive Core */}
                    <div className="space-y-2">
                        <div className="px-2 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-widest">
                            Cognitive Core
                        </div>
                        <div className="space-y-1">
                            <AgentLink
                                icon={Globe}
                                name="Sakana Scientist"
                                active={status?.current_task.includes('Sakana') || status?.current_task.includes('Global')}
                            />
                            <AgentLink
                                icon={Zap}
                                name="InternAgent"
                                active={status?.current_task.includes('Intern')}
                            />
                            <AgentLink
                                icon={ShieldCheck}
                                name="BioDiscovery"
                                active={status?.current_task.includes('Bio')}
                            />
                            <AgentLink
                                icon={Code2}
                                name="CoderAgent"
                                active={status?.current_task.includes('Coder')}
                            />
                        </div>
                    </div>
                </div>
            </ScrollArea>

            <div className="p-4 border-t bg-muted/30">
                <div className="flex items-center gap-3 px-2">
                    <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                        <Beaker className="h-4 w-4 text-primary" />
                    </div>
                    <div className="flex flex-col">
                        <span className="text-xs font-bold leading-none">M3 Ultra</span>
                        <span className="text-[10px] text-muted-foreground">Control Node Alpha</span>
                    </div>
                </div>
            </div>
        </div>
    )
}

function AgentLink({ icon: Icon, name, active }: any) {
    return (
        <div className={`flex items-center justify-between p-2 rounded-md transition-colors ${active ? 'bg-primary/10 text-primary' : 'hover:bg-muted text-muted-foreground hover:text-foreground'}`}>
            <div className="flex items-center gap-3">
                <Icon className={`h-4 w-4 ${active ? 'text-primary' : 'text-muted-foreground'}`} />
                <span className="text-sm font-medium">{name}</span>
            </div>
            {active && <div className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />}
        </div>
    )
}
